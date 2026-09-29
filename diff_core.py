# -*- coding: utf-8 -*-
"""
图片差异提取核心模块 (diff_core)

功能概述
--------
输入两张图片（提取图片 A、背景图片 B），做像素级比较：
只要某个像素的 RGBA 任一通道存在差异（超过容差 tolerance），
该像素就会被提取到输出图中。

双产出
------
每次计算同时产出两张图：

(a) **原色提取图**（K0，不含羽化）——用户要的「不含 alpha 的 png」语义：
  - 有差异的像素 -> alpha = 255，颜色 = 取色来源图片的**原始颜色**；
    若来源图该像素自身携带 alpha（0 < src_alpha < 255），则保留其原始 alpha；
    来源图该像素 alpha == 0 时输出 alpha = 0（不可见，无内容可提取）
  - 无差异的像素 -> alpha = 0（完全透明）
  alpha 只有 0 / 255 两种取值，没有任何半透明过渡。

(b) **羽化素材图层**（K1，feather_radius > 0 时）——可直接放到新背景上的素材：
  在差异掩码**边界外侧**施加羽化，让边缘自然过渡（消除硬边锯齿）。
  掩码内部仍然 alpha=255、颜色 = 来源图原色（恒精确、无杂色）；
  只有边界外侧的「羽化环」带 alpha 渐变。

复合性质（验收标准）
------------------
把输出按 alpha 叠加到底图上：  out*A + (1-out)*B
对不透明源图应当**逐像素精确等于 A 的 RGB**（误差 0）——原色提取图与
羽化图层都满足该性质，因为羽化环的颜色由**精确解族**反解：
     C = B + (A - B) / alpha        （A=来源图外观RGB, B=底图RGB, alpha∈(0,1]）
该式与复合方程 alpha*C + (1-alpha)*B = A 等价，因此任意 alpha 都能精确还原 A。
alpha=255 时它退化为 C=A，与原色提取图完全一致（羽化=0 即二者相同）。

严禁「最小 alpha 解」与掩码模糊
------------------------------
羽化的 alpha **由羽化曲线决定**，颜色再按解族反解；
绝不能反过来去求「使 C 落在色域内的最小 alpha」——那会解出极端颜色（杂色）。
也绝不能对二值掩码做高斯/盒子模糊来得到 alpha：那会把掩码糊成约 51% 的
半透明带（实测 401625 个掩码像素中 205985 个覆盖度落在 (0,1)），
且这些像素的**颜色仍是原色**、与其 alpha 不匹配，叠加回底图根本还原不出
来源图，正是用户报告的"明显锯齿 + 不呈现应有的半透明状态"的根因。

旧的两关键帧 un-blending 路径（_solve_transparency）会让 82% 的可见像素
alpha<64、98.5% 至少一个通道触 0/255、彩度 p90 高达 215，
仅保留为默认关闭的「半透明归因」可选模式（界面勾选后即回到该旧版效果）。

并行优化
--------
图像总像素数较大时（默认 >= 400 万，约 4K 尺寸），使用 multiprocessing
进程池按行切 band 并行计算；小图直接使用单进程 numpy 向量化。

进程池使用 `spawn` 启动方式，避免与 Flask 的多线程环境 fork 造成死锁。
worker 函数 `_band_task` 定义在模块顶层，保证 spawn 子进程可以正常导入。

任何并行路径出现异常都会回退到单进程，并在 stats["engine"] 中标记
"single(fallback)"，绝不让整个请求因并行失败而崩溃。
"""

from __future__ import annotations

import os
import sys
import io
import time
import base64
import argparse
import uuid
import multiprocessing
import concurrent.futures
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

__all__ = [
    "decode_source",
    "compute_keyframes",
    "interpolate_keyframes",
    "render_at",
    "compute_diff",
    "save_png",
    "cleanup_cache",
    "make_preview_dataurl",
    "main",
]

# 触发多进程并行计算的总像素阈值（默认约 4K 尺寸：3840*2160 ≈ 830 万）
DEFAULT_MIN_PIXELS_FOR_MP = 4_000_000

# 边缘羽化默认半径（像素）：对差异掩码**边界外侧**做 alpha 渐变，0 = 不羽化。
# 注意：羽化**不是**对掩码做模糊。模糊会把掩码糊成约 51% 的半透明带，
# 且那些像素颜色仍是原色、与 alpha 不匹配（叠加回底图无法还原来源图）——
# 这正是旧「抗锯齿」选项的缺陷。这里的羽化只作用于掩码外侧，
# 掩码内部恒为 alpha=255 + 原色，外侧按  C = B + (A-B)/alpha  反解颜色。
DEFAULT_FEATHER_RADIUS = 1
MAX_FEATHER_RADIUS = 3

# 四角定位像素：在输出图四角绘制实心标记，便于在 Photoshop 中定位画布位置
DEFAULT_CORNER_MARKERS = True
CORNER_MARKER_SIZE = 4                                  # 标记边长（像素）
CORNER_MARKER_COLOR = (255, 0, 0)                       # 纯红，便于在 PS 中识别和选中

# 「半透明归因」开关（默认 False = 关闭）。
# True 时启用旧的两关键帧 / un-blending 路径（见 _solve_transparency），
# 即最初那版「能输出带 alpha 的透明 PNG、但边缘有杂色」的算法；
# 该路径会对不透明源图做颜色反解，边缘可能出现彩色杂色，仅供按需选用。
DEFAULT_SEMI_ATTRIBUTION = False

# 旧版路径的「透明判定」默认值（0.0-1.0，仅 semi_attribution=True 时有效）：
# 0=差异视为内容变化；1=差异完全归因于透明度；中间值沿精确 alpha 解族插值
DEFAULT_TRANSPARENCY = 0.0

# un-blending 一致性门限（默认关闭）：各通道所需 alpha 的最大分歧超过该值时
# 回退为不透明归因。实测对真实半透明元素（柔和边缘、低 alpha 区域）误伤严重，
# 默认置为 >=1 等效关闭——杂色抑制交给掩码中值滤波精准处理。
SOLVE_SPREAD_TOL = 1.5

# un-blending α 证据下限（默认关闭）：所需 alpha 低于该值时回退为不透明归因。
# 同样会误伤真实的低 alpha 半透明元素，默认 0 关闭；极小 alpha 像素的
# 外扩色位于 ~0 透明度上不可见，由“微差异置空”规则清理。
ALPHA_EVIDENCE_FLOOR = 0.0

# 杂色抑制档位（un-blending 解出颜色的 3x3 掩码中值滤波强度）：
#   0 = 关闭（完全保留逐像素解，噪点放大可见但复合绝对精确）
#   1 = 标准（邻域内容差异 <= 96 才取中值，保护边缘/细线/混沌区）
#   2 = 强  （阈值 150，覆盖放大更厉害的噪点，但更易磨平细节）
DESPECKLE_LEVELS = {0: None, 1: 96.0, 2: 150.0}
DEFAULT_DESPECKLE = 1

# 【不变量】差异掩码内任何真实帧间差 > 2 的像素，alpha 必 >= 1（绝不能是空洞）。
# 数学保证：has_info = any(|d_c| > 2) 为真 -> alpha_min = max(alpha_raw, 1.0) > 0。
# 该不变量由 verify_usercase.py 持续守护——它正是"删除型后处理"必须被否决的原因：
#   speck = alpha*chroma(C) <= 2*alpha*|C-B| = 2*d
# 即"可见彩点强度"永远不超过真实帧间差的两倍，因此凡看得见的彩点背后
# 都是真实内容，删掉它必然在叠加回原底图时留下同样量级的空洞。

# 预览图默认最长边
DEFAULT_PREVIEW_SIZE = 480

# 预览图合成用的深灰色底（含 alpha 的图先贴到这个底色上）
PREVIEW_BG_COLOR = (0x1F, 0x24, 0x30)


# ---------------------------------------------------------------------------
# 图片解码
# ---------------------------------------------------------------------------
def decode_source(src: dict) -> Image.Image:
    """
    将前端传入的图片描述解码为 PIL RGBA 图像。

    参数
    ----
    src : dict
        {"type": "path", "value": "E:\\\\a.png"}
        或 {"type": "data", "value": "data:image/png;base64,...."}

    返回
    ----
    PIL.Image.Image  (mode = "RGBA")

    异常
    ----
    ValueError : 参数不合法、文件不存在、base64 损坏、图片无法解码等，
                 消息全部为中文，可直接展示给用户。
    """
    if not isinstance(src, dict):
        raise ValueError("图片参数格式错误：应为包含 type 与 value 的对象")

    stype = src.get("type")
    value = src.get("value")

    if stype not in ("path", "data"):
        raise ValueError("图片参数错误：type 必须是 'path' 或 'data'")

    if not isinstance(value, str) or not value.strip():
        raise ValueError("图片参数错误：value 不能为空")

    value = value.strip()

    if stype == "path":
        return _decode_from_path(value)
    return _decode_from_dataurl(value)


def _decode_from_path(raw_path: str) -> Image.Image:
    """按本地文件路径读取图片（支持 Windows 与 WSL 路径）。"""
    try:
        p = Path(raw_path).expanduser()
    except Exception as exc:  # 路径本身非法（例如含有非法字符）
        raise ValueError(f"路径无法解析：{raw_path}（{exc}）") from exc

    if not p.exists():
        raise ValueError(f"文件不存在：{p}")
    if not p.is_file():
        raise ValueError(f"路径不是一个文件：{p}")

    try:
        with Image.open(p) as im:
            # GIF 等多帧图片取第一帧；EXIF 方向统一应用，避免"看起来一样却判为不同"
            try:
                im.seek(0)
            except Exception:
                pass
            im = ImageOps.exif_transpose(im)
            return im.convert("RGBA")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"无法解码图片（文件可能损坏或格式不支持）：{p}（{exc}）") from exc


def _decode_from_dataurl(value: str) -> Image.Image:
    """解码 data URL（data:image/png;base64,....）或纯 base64 字符串。"""
    payload = value
    if payload.startswith("data:"):
        # 去掉 "data:image/png;base64," 前缀
        comma = payload.find(",")
        if comma < 0:
            raise ValueError("图片数据格式错误：data URL 缺少逗号分隔符")
        payload = payload[comma + 1:]

    # 去掉可能存在的空白字符（换行等）
    payload = "".join(payload.split())

    if not payload:
        raise ValueError("图片数据为空")

    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise ValueError(f"Base64 解码失败：图片数据可能已损坏（{exc}）") from exc

    if not raw:
        raise ValueError("Base64 解码后数据为空")

    try:
        with Image.open(io.BytesIO(raw)) as im:
            try:
                im.seek(0)
            except Exception:
                pass
            im = ImageOps.exif_transpose(im)
            return im.convert("RGBA")
    except Exception as exc:
        raise ValueError(f"无法解码图片数据（格式不支持或内容已损坏）：{exc}") from exc


# ---------------------------------------------------------------------------
# 差异计算
# ---------------------------------------------------------------------------
def _band_task(a_band: np.ndarray, b_band: np.ndarray, tolerance: int,
               source_is_a: bool):
    """
    单个 band 的差异计算（必须是模块顶层函数，spawn 子进程才能导入）。

    参数
    ----
    a_band, b_band : np.ndarray  uint8, 形状 (h, w, 4)，已保证尺寸一致
    tolerance      : int  容差 0-255
    source_is_a    : bool True 时颜色取自 A，否则取自 B

    返回
    ----
    (rgb_band, mask_band, src_alpha_band)
      rgb_band       : np.ndarray uint8 (h, w, 3) 取色来源的 RGB
      mask_band      : np.ndarray bool  (h, w)    该像素是否存在差异（二值）
      src_alpha_band : np.ndarray uint8 (h, w)    取色来源像素的 alpha
                       （保留半透明信息，供输出按源透明度合成）

    半透明严谨判定
    -------------
    alpha == 0   -> 全透明：该像素不可见，其 RGB 无定义
    alpha == 255 -> 不透明
    0 < a < 255  -> 半透明
    两张图在该像素都全透明时视为相同（RGB 不参与比较），
    避免“透明区域 RGB  garbage 不同”造成的误报。
    """
    # int16 差值：uint8 相减会溢出，必须先升位
    diff = np.abs(a_band.astype(np.int16) - b_band.astype(np.int16)).max(axis=2)

    a_alpha = a_band[..., 3]
    b_alpha = b_band[..., 3]
    both_transparent = (a_alpha == 0) & (b_alpha == 0)

    mask = (diff > int(tolerance)) & ~both_transparent

    rgb = a_band[..., :3] if source_is_a else b_band[..., :3]
    src_alpha = a_alpha if source_is_a else b_alpha
    pick_rgb = a_band[..., :3] if source_is_a else b_band[..., :3]
    return rgb, mask, src_alpha, pick_rgb


def _split_bands(height: int, band_count: int):
    """把 height 行尽量均匀切成 band_count 段，返回 [(y0, y1), ...]。"""
    band_count = max(1, min(int(band_count), int(height)))
    edges = np.linspace(0, height, band_count + 1).astype(int)
    bands = []
    for i in range(band_count):
        y0 = int(edges[i])
        y1 = int(edges[i + 1])
        if y1 > y0:  # 高度至少 1 行
            bands.append((y0, y1))
    return bands


def _band_worker(payload):
    """进程池任务包装：接收 (a_band, b_band, tolerance, source_is_a)。"""
    a_band, b_band, tolerance, source_is_a = payload
    return _band_task(a_band, b_band, tolerance, source_is_a)


# ---------------------------------------------------------------------------
# 边缘抗锯齿（旧工具，默认路径已不用）
# ---------------------------------------------------------------------------
# 说明：默认提取路径已改用「边缘羽化」（_apply_feather），不再对掩码做模糊——
# 模糊会把掩码糊成约 51% 的半透明带且颜色与 alpha 不匹配，是块状杂色的来源。
# 下面两个函数仅作为通用工具保留（离线校验脚本与第三方调用仍会引用），
# 不再参与 compute_keyframes 的默认流程。
def _blur_axis(f: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """沿单个轴做一次盒子模糊（滑动窗口均值），边缘用 edge 填充。"""
    win = 2 * radius + 1
    if axis == 1:
        p = np.pad(f, ((0, 0), (radius, radius)), mode="edge")
        cs = np.cumsum(p, axis=1, dtype=np.float32)
        cs = np.concatenate([np.zeros((cs.shape[0], 1), np.float32), cs], axis=1)
        return (cs[:, win:] - cs[:, :-win]) / float(win)
    p = np.pad(f, ((radius, radius), (0, 0)), mode="edge")
    cs = np.cumsum(p, axis=0, dtype=np.float32)
    cs = np.concatenate([np.zeros((1, cs.shape[1]), np.float32), cs], axis=0)
    return (cs[win:, :] - cs[:-win, :]) / float(win)


def _antialias_alpha(alpha_bin: np.ndarray, radius: int) -> np.ndarray:
    """对二值 alpha 掩码做边缘抗锯齿（3 次可分离盒子模糊 ≈ 高斯）。"""
    if radius <= 0:
        return alpha_bin
    f = alpha_bin.astype(np.float32) * (1.0 / 255.0)
    for _ in range(3):
        f = _blur_axis(f, radius, 1)
        f = _blur_axis(f, radius, 0)
    return np.clip(f * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 边缘羽化（alpha 渐变 + 颜色按精确解族反解）
# ---------------------------------------------------------------------------
def _box_dilate(mask: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """
    对 mask[y0:y1] 做 3x3（8 邻域）膨胀，读邻居时越过 band 边界读整图。

    逐 band 计算是羽化的关键：羽化需要掩码外侧邻域的信息，
    直接在整图上做多次膨胀会重复读写整幅 (h,w) 布尔数组（实测 4 次约 21ms）。
    按 band 处理时每个 band 只读写自己那几行，缓存友好。
    """
    h, w = mask.shape
    ya = y0 - 1 if y0 > 0 else y0
    yb = y1 + 1 if y1 < h else y1
    src = mask[ya:yb]
    top = y0 - ya          # 中心行在 src 中的起点
    bot = top + (y1 - y0)
    out = np.zeros((y1 - y0, w), dtype=bool)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            # 邻居切片在 src 中的行范围 [top+dy, bot+dy)
            r0 = max(top + dy, 0)
            r1 = min(bot + dy, src.shape[0])
            if r1 <= r0:
                continue
            c0 = max(dx, 0)
            c1 = min(w + dx, w)
            if c1 <= c0:
                continue
            piece = src[r0:r1, c0:c1]
            o0 = r0 - (top + dy)
            o1 = o0 + piece.shape[0]
            out[o0:o1, c0 - dx:c1 - dx] |= piece
    return out


def _feather_distance(mask: np.ndarray, radius: int,
                      band_rows: int = 0) -> np.ndarray:
    """
    计算每个像素到掩码的 8 邻域（切比雪夫）距离，掩码内部记 0。

    迭代膨胀 radius 次：每轮把「已覆盖集合」向外扩一圈，
    新扩出来的像素距离 = 轮次。返回 int32 距离图（仅 0..radius 有值）。

    参数
    ----
    mask      : np.ndarray bool (h, w) 二值差异掩码
    radius    : int  羽化半径（迭代轮数），<=0 时返回全 0
    band_rows : int  <=0 时整图一次算；>0 时按该行数分块，
                     每块只对必要行做膨胀（见 _box_dilate）

    返回
    ----
    np.ndarray int32 (h, w)
    """
    h, w = mask.shape
    dist = np.zeros((h, w), dtype=np.int32)
    if radius <= 0:
        return dist

    # covered：已经确定过距离的像素（含掩码本身）
    covered = mask.copy()
    frontier = mask  # 上一轮新确定的像素，本轮从它向外扩

    if band_rows and band_rows < h:
        for k in range(1, radius + 1):
            for y0 in range(0, h, band_rows):
                y1 = min(h, y0 + band_rows)
                nb = _box_dilate(frontier, y0, y1)
                blk = nb & ~covered[y0:y1]
                dist[y0:y1][blk] = k
            covered |= dist > 0
            covered |= mask
            frontier = (dist == k)
    else:
        for k in range(1, radius + 1):
            p = np.pad(frontier, 1, mode="constant", constant_values=False)
            nb = np.zeros((h, w), dtype=bool)
            for dy in range(3):
                for dx in range(3):
                    if dy == 1 and dx == 1:
                        continue
                    nb |= p[dy:dy + h, dx:dx + w]
            new = nb & ~covered
            dist[new] = k
            covered |= new
            frontier = new

    return dist


def _apply_feather(k0: np.ndarray, mask: np.ndarray, s_rgb: np.ndarray,
                   o_rgb: np.ndarray, radius: int,
                   src_alpha: np.ndarray | None = None) -> np.ndarray:
    """
    由原色提取图 K0 生成羽化素材图层。

    羽化曲线（半径 R，距离 d ∈ [1, R]）：
        alpha = 1 - d / (R + 1)
    R=1 -> 0.500；R=2 -> 0.333/0.667；R=3 -> 0.250/0.500/0.750。
    掩码内部（d=0）恒为 alpha=1（255），颜色 = 来源图原色。

    颜色反解（精确解族）
    ------------------
        C = B + (A - B) / alpha
    与复合方程 alpha*C + (1-alpha)*B = A 等价，故任意 alpha 都精确还原 A。
    alpha 越是接近 1，反解越稳定；alpha 很小时 (A-B)/alpha 会放大到色域外，
    此时**钳住 alpha 的下限**而不是裁剪颜色——裁剪会引入色偏（彩块），
    而抬高 alpha 只需把 C 拉回色域边缘，复合结果仍精确。

    alpha 下限有闭式解（使三通道 C 全部落在 [0,255] 的最小 alpha）：
        A >= B 的通道：  alpha >= (A - B) / (255 - B)
        A <  B 的通道：  alpha >= (B - A) / B
    这正是取「最小 alpha 解」—但**仅用于钳住下限**，颜色始终由曲线给定的
    alpha 反解；绝不会采用那个解出来的极端颜色（那才是杂色来源）。

    与来源图自身 alpha 的关系：源像素半透明时时，其"外观色"就是
    折算后的 A；K0 在该像素的 alpha 是 src_alpha，羽化按比例缩放它，
    颜色仍按同一解族反解，叠加回底图后等于来源图**覆盖后的外观**。

    参数
    ----
    k0     : np.ndarray uint8 (h, w, 4) 原色提取图（羽化=0 的结果）
    mask   : np.ndarray bool  (h, w)    二值差异掩码
    s_rgb  : np.ndarray uint8 (h, w, 3) 来源图 RGB
    o_rgb  : np.ndarray uint8 (h, w, 3) 底图 RGB
    radius : int  羽化半径

    返回
    ----
    np.ndarray uint8 (h, w, 4)
    """
    if radius <= 0:
        return k0.copy()

    h, w = mask.shape
    dist = _feather_distance(mask, radius, band_rows=_feather_band_rows(h, w))

    # 羽化曲线
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[mask] = 1.0
    ring = (dist >= 1) & (dist <= radius)
    if ring.any():
        d = dist[ring].astype(np.float32)
        alpha[ring] = 1.0 - d / float(radius + 1)

    # 源像素自身 alpha 参与缩放（不透明源图时恒为 1.0，不改变 alpha）。
    # 注意必须用**源图自身的 alpha 通道**（src_alpha），不能用 k0 的输出 alpha——
    # 羽化环位于掩码外侧，k0 在那里 alpha=0，若用它缩放会把整圈环乘成 0（羽化失效）。
    if src_alpha is None:
        src_a = np.ones((h, w), dtype=np.float32)
    else:
        src_a = src_alpha.astype(np.float32) * (1.0 / 255.0)
    alpha = alpha * src_a

    out = k0.copy()
    if not ring.any():
        return out

    S = s_rgb[ring].astype(np.float32)
    O = o_rgb[ring].astype(np.float32)
    a = alpha[ring].astype(np.float32)

    # ---- alpha 下限（闭式解）：保证 C = O + (S-O)/a 不越界 ----
    diff = S - O
    pos = diff >= 0.0
    denom_pos = np.where(O < 255.0, 255.0 - O, 1.0)
    denom_neg = np.where(O > 0.0, O, 1.0)
    req = np.where(pos, diff / denom_pos, (-diff) / denom_neg)
    req = np.where(diff == 0.0, 0.0, req)
    floor = req.max(axis=1)                      # 每像素的最小可行 alpha

    a_safe = np.maximum(a, floor)
    # 下限本身可能 > 1（S 与 O 极端差异时无解），此时 alpha 只能取 1
    a_safe = np.clip(a_safe, 1e-6, 1.0)

    C = O + (S - O) / a_safe[:, None]
    C = np.clip(C, 0.0, 255.0)                   # 兜底：正常路径下不会触发

    out[ring, :3] = np.round(C).astype(np.uint8)
    out[ring, 3] = np.round(a_safe * 255.0).astype(np.uint8)
    return out


def _feather_band_rows(h: int, w: int) -> int:
    """羽化分块行数：按面积把每个 band 控制在约 400 万像素以内。"""
    if h <= 0 or w <= 0:
        return 0
    rows = int(4_000_000 // max(1, w))
    return max(64, min(h, rows))


# ---------------------------------------------------------------------------
# 四角定位像素
# ---------------------------------------------------------------------------
def _masked_median3(values: np.ndarray, mask: np.ndarray, min_support: int = 5,
                    uniform_tol: float = 96.0) -> np.ndarray:
    """
    3x3 掩码中值滤波（带一致性条件）。

    仅当同时满足以下条件时才用邻域中值替换中心像素：
      1. 中心像素属于 mask；
      2. 邻域中属于 mask 的像素数 >= min_support；
      3. 这些像素的 values 差异（max-min）<= uniform_tol（邻域内容均匀）。
    任一不满足则保持原值——保护边缘、细线、孤立点，以及内容混沌的区域
    （混沌区的中值没有意义，其原始解本就是精确的）。

    用途：un-blending 求解 C = O + d/α 会把噪点差异放大 1/α 倍
    （半透明元素的抗锯齿边缘 α 很小，放大最厉害），在元素内部/边缘
    形成彩色斑点（杂色）。对内容均匀的邻域取中值可还原真实颜色。
    """
    h, w = mask.shape
    if h < 3 or w < 3 or values.shape[:2] != (h, w):
        return values
    ch = values.shape[2]
    out = values.copy()
    mp = np.pad(mask, 1, mode="constant", constant_values=False)
    vp = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="edge")
    BIG = np.float32(1e9)
    # 按行块处理，控制内存（9 个邻域的栈）
    block = max(1, int(96e6 / max(1.0, 9 * w * ch * 4)))
    for y0 in range(0, h, block):
        y1 = min(h, y0 + block)
        bh = y1 - y0
        stack = np.empty((9, bh, w, ch), np.float32)
        cnt = np.zeros((bh, w), np.int32)
        k = 0
        for dy in range(3):
            for dx in range(3):
                m = mp[y0 + dy:y0 + dy + bh, dx:dx + w]
                stack[k] = np.where(m[..., None], vp[y0 + dy:y0 + dy + bh, dx:dx + w], BIG)
                cnt += m
                k += 1
        stack.sort(axis=0)
        idx_max = np.maximum(cnt - 1, 0)             # 最大有效值的下标
        idx_med = idx_max // 2                       # 下中位数的下标
        med = np.take_along_axis(stack, idx_med[None, ..., None].astype(np.intp), axis=0)[0]
        vmin = stack[0]                              # 排序后索引 0 即最小值（无效值排在末尾）
        vmax = np.take_along_axis(stack, idx_max[None, ..., None].astype(np.intp), axis=0)[0]
        spread = (vmax - vmin).max(axis=-1)
        use = (cnt >= min_support) & (spread <= uniform_tol) & mask[y0:y1]
        out[y0:y1][use] = med[use]
    return out


def _draw_corner_markers(out: np.ndarray) -> None:
    """
    在输出图四个角绘制实心定位像素（纯红、alpha=255），便于在 PS 中定位。

    标记紧贴画布边缘，因此选中标记即可确定输出图的精确边界；
    标记不参与差异统计，也不受不透明度滑块影响（始终实心）。
    原地修改 out。
    """
    h, w = out.shape[:2]
    ms = min(CORNER_MARKER_SIZE, w, h)
    if ms <= 0:
        return
    r, g, b = CORNER_MARKER_COLOR
    for y0 in (0, h - ms):
        for x0 in (0, w - ms):
            out[y0:y0 + ms, x0:x0 + ms, 0] = r
            out[y0:y0 + ms, x0:x0 + ms, 1] = g
            out[y0:y0 + ms, x0:x0 + ms, 2] = b
            out[y0:y0 + ms, x0:x0 + ms, 3] = 255


def _normalize_canvas(a_img: Image.Image, b_img: Image.Image):
    """
    尺寸归一化：以两图宽高最大值建透明画布，较小的图用透明像素补齐。

    返回 (a_rgba, b_rgba, width, height, padded)
    padded 表示是否发生了补齐（补齐区域天然算作差异）。
    """
    a_rgba = a_img.convert("RGBA")
    b_rgba = b_img.convert("RGBA")

    wa, ha = a_rgba.size
    wb, hb = b_rgba.size
    w = max(wa, wb)
    h = max(ha, hb)

    padded = (wa != wb) or (ha != hb)

    if wa != w or ha != h:
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.paste(a_rgba, (0, 0))
        a_rgba = canvas

    if wb != w or hb != h:
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.paste(b_rgba, (0, 0))
        b_rgba = canvas

    return a_rgba, b_rgba, w, h, padded


def _bbox_of_mask(alpha: np.ndarray):
    """计算 alpha>0 像素的外接框，返回 [x0, y0, x1, y1]（x1/y1 为闭区间）或 None。"""
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any():
        return None
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    return [int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])]


def _solve_transparency(s_rgb: np.ndarray, o_rgb: np.ndarray,
                        src_alpha: np.ndarray | None = None):
    """
    将「来源图覆盖在另一张图上」造成的像素差异归因于透明度（un-blending）。

    对每个像素求解 (alpha, C)，使   alpha * C + (1 - alpha) * O = S_eff
    逐通道成立，其中 O 为底图像素 RGB，S_eff 为来源图**覆盖在底图上的外观**：
    来源图自身带 alpha 时先折算（s_eff = a*S + (1-a)*O），否则 S_eff = S。
    （半透明来源图的 RGB 并非其外观，直比较会高估差异、解出错误的透明度。）

    数学推导
    --------
    单通道方程在两个未知数 (alpha, C) 下欠定；取使 C 仍落在 [0,255]
    色域内的最小 alpha（此时 C 在某一通道恰好触界）：
        S >= O 时  alpha >= (S - O) / (255 - O)
        S <  O 时  alpha >= (O - S) / O
    像素 alpha = 各通道要求的最大值；代回即得 C。

    一致性门限
    ----------
    真半透明在各通道所需的 alpha 大致相等；若分歧明显（> 0.25），
    说明该像素差异无法用均匀透明度解释（噪点、非均匀变化），
    回退为不透明归因 alpha=1, C=S_eff——该解永远精确且不放大噪点。

    参数
    ----
    s_rgb, o_rgb : np.ndarray uint8 (h, w, 3)
    src_alpha    : np.ndarray uint8 (h, w) 或 None，来源图像素 alpha

    返回
    ----
    alpha_min  : np.ndarray float32 (h, w)，值域 [0,1]
    c_solved   : np.ndarray float32 (h, w, 3)
    consistent : np.ndarray bool (h, w)，该像素是否通过一致性门限
    """
    S = s_rgb.astype(np.float32)
    O = o_rgb.astype(np.float32)
    if src_alpha is not None:  # 来源图覆盖在底图上的外观
        a = src_alpha.astype(np.float32) * (1.0 / 255.0)
        S = S * a[..., None] + O * (1.0 - a[..., None])

    d = S - O
    pos = d >= 0
    denom_pos = np.where(O < 255.0, 255.0 - O, 1.0)   # O=255 时 d 必 <=0，不会用到
    denom_neg = np.where(O > 0.0, O, 1.0)             # O=0 时 d 必 >=0，不会用到
    req = np.where(pos, d / denom_pos, (-d) / denom_neg)
    req = np.where(d == 0.0, 0.0, req)                # 无差异通道不提出要求
    alpha_raw = req.max(axis=2)

    # 一致性门限：仅在“有信息量”的通道（差异>2）之间比较需求分歧
    informative = np.abs(d) > 2.0
    has_info = informative.any(axis=2)
    req_hi = np.where(informative, req, -1.0).max(axis=2)
    req_lo = np.where(informative, req, 2.0).min(axis=2)
    spread = np.where(has_info, req_hi - req_lo, 0.0)
    # 分歧过大（无法用均匀透明度解释）或 alpha 证据不足（外扩噪点太强）时，
    # 回退为不透明归因 alpha=1, C=外观色——该解永远精确且不放大噪点
    consistent = (spread <= SOLVE_SPREAD_TOL) & (
        (alpha_raw >= ALPHA_EVIDENCE_FLOOR) | ~has_info)

    alpha_min = np.where(consistent, alpha_raw, 1.0)
    # 差异可忽略（所有通道 |d|<=2）的像素：没有可归因的透明度，输出为空
    # （否则 1-2 级差会被外放成饱和色，虽是透明不可见但会污染文件）
    alpha_min = np.where(has_info, alpha_min, 0.0)

    am = alpha_min[..., None]
    am_safe = np.where(am > 1e-6, am, 1.0)            # 防 0 除（无差异像素的结果不会被使用）
    c = np.clip((S - (1.0 - am_safe) * O) / am_safe, 0.0, 255.0)
    # 回退通道：C 取外观色（与 alpha=1 配合永远精确）
    c = np.where(consistent[..., None], c, S)
    return alpha_min, c, consistent


def compute_keyframes(a_img: Image.Image, b_img: Image.Image, tolerance: int = 0,
                      source: str = "a", workers: int | None = None,
                      min_pixels_for_mp: int = DEFAULT_MIN_PIXELS_FOR_MP,
                      feather_radius: int = DEFAULT_FEATHER_RADIUS,
                      despeckle: int = DEFAULT_DESPECKLE,
                      semi_attribution: bool = DEFAULT_SEMI_ATTRIBUTION):
    """
    计算两张图片的像素级差异，返回两个「关键帧」与统计信息。

    关键帧
    ------
    K0（默认，精确合成提取）：差异像素 = 取色来源的原色，alpha = 255；
        来源像素自身半透明（0 < src_alpha < 255）时 alpha 取 src_alpha；
        来源像素全透明（src_alpha == 0）时 alpha = 0。相同像素 alpha = 0。
        该结果叠加到底图上逐像素精确等于来源图（数学恒等，无误差）。
    K1（仅半透明归因开启时有效）：un-blending 求解每像素 (alpha, C)，
        使提取结果叠加到底图上仍还原来源图（旧版算法，边缘可能有杂色）。

    semi_attribution=False（默认）时 k1 直接复用 k0（无需第二遍计算，
    「透明判定」滑块与插值渲染在界面中已随之移除，节省一份 PNG 编码与传输）。

    参数
    ----
    a_img, b_img      : PIL.Image.Image  两张输入图片（任意 PIL 可读格式）
    tolerance         : int   容差 0-255，任一通道差值 > tolerance 才算差异
    source            : str   "a" 或 "b"，差异像素的颜色取自哪张图
    workers           : int|None  None 表示自动（os.cpu_count()）；<=1 强制单进程
    min_pixels_for_mp : int   达到该总像素数才启用多进程
    feather_radius    : int   边缘羽化半径（像素），0 = 不羽化（硬边）
    despeckle         : int   杂色抑制档位 0/1/2（仅半透明归因路径使用）
    semi_attribution  : bool  是否启用「半透明归因」旧版算法（默认 False）

    返回
    ----
    (k0: PIL.Image.Image, k1: PIL.Image.Image, stats: dict)
    """
    t_start = time.perf_counter()

    if a_img is None or b_img is None:
        raise ValueError("输入图片不能为空")

    # 参数规整
    try:
        tolerance = int(tolerance)
    except Exception:
        raise ValueError("容差 tolerance 必须是整数（0-255）")
    if tolerance < 0 or tolerance > 255:
        raise ValueError("容差 tolerance 必须在 0-255 之间")

    try:
        feather_radius = int(feather_radius)
    except Exception:
        feather_radius = DEFAULT_FEATHER_RADIUS
    feather_radius = max(0, min(MAX_FEATHER_RADIUS, feather_radius))

    source_is_a = str(source).strip().lower() != "b"  # 默认取 A

    # 尺寸归一化
    a_rgba, b_rgba, w, h, padded = _normalize_canvas(a_img, b_img)

    A = np.asarray(a_rgba, dtype=np.uint8)
    B = np.asarray(b_rgba, dtype=np.uint8)

    total_pixels = int(w) * int(h)

    # 决定进程数
    if workers is None:
        workers = os.cpu_count() or 1
    try:
        workers = int(workers)
    except Exception:
        workers = os.cpu_count() or 1
    if workers <= 0:
        workers = os.cpu_count() or 1
    workers = max(1, workers)

    use_mp = (workers > 1) and (total_pixels >= int(min_pixels_for_mp)) and h > 1

    mask_full = np.zeros((h, w), dtype=bool)           # 二值差异掩码（统计口径）
    src_alpha_full = np.zeros((h, w), dtype=np.uint8)  # 取色来源像素的 alpha
    pick_rgb_full = np.zeros((h, w, 3), dtype=np.uint8)  # 取色来源像素的原始 RGB
    engine = "single"
    actual_workers = 1

    if use_mp:
        # band 数 = 进程数 * 2，便于负载均衡；行数不足时自动减少
        band_count = min(workers * 2, h)
        bands = _split_bands(h, band_count)
        band_count = len(bands)
        actual_workers = max(1, min(workers, band_count))

        try:
            ctx = multiprocessing.get_context("spawn")
            payloads = [
                (A[y0:y1], B[y0:y1], tolerance, source_is_a)
                for (y0, y1) in bands
            ]
            results = [None] * len(bands)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=actual_workers, mp_context=ctx
            ) as pool:
                futures = {
                    pool.submit(_band_worker, p): idx
                    for idx, p in enumerate(payloads)
                }
                for fut in concurrent.futures.as_completed(futures):
                    idx = futures[fut]
                    results[idx] = fut.result()  # 子进程异常在此抛出

            for idx, (y0, y1) in enumerate(bands):
                _rgb_band, mask_band, alpha_band, pick_band = results[idx]
                mask_full[y0:y1] = mask_band
                src_alpha_full[y0:y1] = alpha_band
                pick_rgb_full[y0:y1] = pick_band

            engine = "multiprocess"

        except Exception as exc:
            # 并行失败：回退单进程，绝不让请求崩溃
            print(f"[diff_core] 多进程计算失败，回退单进程：{exc}", file=sys.stderr)
            mask_full = np.zeros((h, w), dtype=bool)
            src_alpha_full = np.zeros((h, w), dtype=np.uint8)
            pick_rgb_full = np.zeros((h, w, 3), dtype=np.uint8)
            engine = "single(fallback)"
            actual_workers = 1

    if engine != "multiprocess":
        # 单进程（含多进程失败回退）：numpy 向量化整图一次计算
        _rgb, mask, src_alpha, pick_rgb = _band_task(A, B, tolerance, source_is_a)
        mask_full = mask
        src_alpha_full = src_alpha
        pick_rgb_full = pick_rgb
        actual_workers = 1

    # ---- 统计（基于二值差异掩码，不含抗锯齿过渡带） ----
    diff_pixels = int(np.count_nonzero(mask_full))
    ratio = round(float(diff_pixels) / float(total_pixels), 4) if total_pixels else 0.0
    bbox = _bbox_of_mask(mask_full)

    # ---- 来源图（S）与底图（O）的 RGB ----
    s_rgb = A[..., :3] if source_is_a else B[..., :3]
    o_rgb = B[..., :3] if source_is_a else A[..., :3]

    # ---- K0：原色提取图（不含羽化，无任何颜色反解） ----
    # 差异像素 alpha = 255，颜色 = 取色来源原色 —— 叠加到底图时
    #   out*A + (1-out)*B = 1*颜色 + 0*B = 来源原色
    # 逐像素恒等，与差异大小无关，不可能产生杂色。
    # 源像素自身 alpha == 0 -> 无内容可提取，输出 alpha = 0（不可见）；
    # 源像素半透明 -> 保留其原始 alpha；其余 -> 255。
    # 注意：这里**不做**任何掩码模糊。旧「抗锯齿」会把掩码糊成约 51% 的
    # 半透明带，且那些像素颜色仍是原色、与 alpha 不匹配（叠加回底图无法
    # 还原来源图）——那正是边缘块状/杂色的来源。边缘过渡改由 K1 羽化负责。
    k0 = np.zeros((h, w, 4), dtype=np.uint8)
    k0[..., :3] = pick_rgb_full
    k0[..., 3] = np.where(mask_full, src_alpha_full, np.uint8(0)).astype(np.uint8)

    if not semi_attribution:
        # 默认路径：K1 = 羽化素材图层。在差异掩码**边界外侧**做 alpha 渐变，
        # 颜色按精确解族反解，因此换到新背景上边缘自然过渡（无硬边锯齿），
        # 而叠回原底图仍逐像素还原来源图。feather_radius = 0 时 K1 == K0。
        k1_arr = (_apply_feather(k0, mask_full, s_rgb, o_rgb, feather_radius,
                                 src_alpha_full)
                  if feather_radius > 0 else k0.copy())
        return (Image.fromarray(k0, "RGBA"), Image.fromarray(k1_arr, "RGBA"),
                _build_stats(
                    w, h, diff_pixels, total_pixels, ratio, bbox, engine,
                    actual_workers, t_start, source_is_a, tolerance,
                    feather_radius, despeckle, padded,
                    src_alpha_full, mask_full, a_img, b_img,
                    semi_attribution=False, transparency=None,
                ))

    # ---- K1：差异完全归因于透明度（un-blending，旧版算法） ----
    # 注意：仅对真实半透明内容才成立；对不透明源图会解出「极小 alpha + 极端颜色」，
    # 叠加虽精确但视觉上产生严重杂色，故默认关闭。
    alpha_min, c_solved, consistent = _solve_transparency(s_rgb, o_rgb, src_alpha_full)
    level = int(despeckle)
    if level not in DESPECKLE_LEVELS:
        level = DEFAULT_DESPECKLE
    c_clean = c_solved.astype(np.float32)
    k1_alpha_u8 = np.round(alpha_min * mask_full * 255.0).astype(np.uint8)
    if level > 0:
        c_candidate = _masked_median3(
            c_clean, mask_full & consistent,
            min_support=5,
            uniform_tol=DESPECKLE_LEVELS[level],
        )
        a1f = k1_alpha_u8.astype(np.float32) / 255.0
        delta_c = np.max(np.abs(c_candidate - c_clean), axis=2)
        allowed = (k1_alpha_u8 > 0) & (a1f * delta_c <= 2.0)
        c_clean = np.where(allowed[..., None], c_candidate, c_clean)
    k1 = np.zeros((h, w, 4), dtype=np.uint8)
    k1[..., :3] = np.round(np.clip(c_clean, 0.0, 255.0)).astype(np.uint8)
    k1[..., 3] = k1_alpha_u8

    stats = _build_stats(
        w, h, diff_pixels, total_pixels, ratio, bbox, engine, actual_workers,
        t_start, source_is_a, tolerance, feather_radius, despeckle, padded,
        src_alpha_full, mask_full, a_img, b_img,
        semi_attribution=True, transparency=DEFAULT_TRANSPARENCY,
    )
    return Image.fromarray(k0, "RGBA"), Image.fromarray(k1, "RGBA"), stats


def _build_stats(w, h, diff_pixels, total_pixels, ratio, bbox, engine,
                 actual_workers, t_start, source_is_a, tolerance, feather_radius,
                 despeckle, padded, src_alpha_full, mask_full, a_img, b_img,
                 semi_attribution, transparency):
    """组装统计信息（两条路径共用）。"""
    # ---- 差异像素的透明度构成（严谨判定：0=全透明 / 255=不透明 / 其余=半透明） ----
    if diff_pixels:
        src_a = src_alpha_full[mask_full]
        opaque_count = int(np.count_nonzero(src_a == 255))
        semi_count = int(np.count_nonzero((src_a > 0) & (src_a < 255)))
        transparent_count = int(np.count_nonzero(src_a == 0))
    else:
        opaque_count = semi_count = transparent_count = 0

    elapsed_ms = int(round((time.perf_counter() - t_start) * 1000))

    return {
        "width": int(w),
        "height": int(h),
        "diff_pixels": diff_pixels,
        "total_pixels": total_pixels,
        "ratio": ratio,
        "bbox": bbox,
        "engine": engine,
        "workers": int(actual_workers),
        "elapsed_ms": elapsed_ms,
        "source": "a" if source_is_a else "b",
        "tolerance": int(tolerance),
        "feather_radius": int(feather_radius),
        "despeckle": int(despeckle),
        "padded": bool(padded),
        "semi_attribution": bool(semi_attribution),
        "transparency": (None if transparency is None else round(float(transparency), 4)),
        "alpha_breakdown": {
            "opaque": opaque_count,
            "semi": semi_count,
            "transparent": transparent_count,
        },
        "a_size": [int(a_img.size[0]), int(a_img.size[1])],
        "b_size": [int(b_img.size[0]), int(b_img.size[1])],
    }


# ---------------------------------------------------------------------------
# 透明判定渲染（关键帧插值）
# ---------------------------------------------------------------------------
def make_render_auxiliary(a_img: Image.Image, b_img: Image.Image,
                          source: str = "a") -> tuple[Image.Image, Image.Image]:
    """返回精确 alpha 解族所需的来源外观 S 与底图 O 两张 RGB 图。"""
    A, B, _w, _h, _padded = _normalize_canvas(a_img, b_img)
    aa = np.asarray(A, dtype=np.uint8)
    bb = np.asarray(B, dtype=np.uint8)
    if str(source).strip().lower() == "b":
        src, other = bb, aa
    else:
        src, other = aa, bb
    s = src[..., :3].astype(np.float32)
    o = other[..., :3].astype(np.float32)
    alpha = src[..., 3:4].astype(np.float32) / 255.0
    s_eff = np.round(s * alpha + o * (1.0 - alpha)).astype(np.uint8)
    return Image.fromarray(s_eff, "RGB"), Image.fromarray(o.astype(np.uint8), "RGB")


def interpolate_keyframes(k0: Image.Image, k1: Image.Image, t: float,
                           source_effective: Image.Image | None = None,
                           background: Image.Image | None = None) -> Image.Image:
    """
    按透明判定 t 渲染两端关键帧之间的结果。

    提供 sidecar（来源外观 S_eff 与底图 O）时走「精确 alpha 解族」：

        alpha(t) = (1-t)*alpha0 + t*alpha1
        C(t)     = O + (S_eff - O) / alpha(t)

    解族中任意 alpha 都满足 alpha*C + (1-alpha)*O = S_eff，
    因此提取层叠加到底图后能精确还原来源图（8bit 量化误差 <=2），
    且与 t 的取值无关。直接对 RGBA 做线性插值会让颜色偏离解族，
    在 t=0.5 处产生数十级的复合偏差（饱和度高的差异像素尤其严重）。

    无 sidecar 时退化为 RGBA 线性插值（兼容旧调用方）。

    取整统一为 floor(x + 0.5)（半值向上），与前端 JavaScript 实现逐位一致，
    保证同一 t 下预览与导出结果完全相同。全部运算使用 float64，
    与 JavaScript 的 Number 运算位级相同。
    """
    t = min(1.0, max(0.0, float(t)))
    if t <= 0.0:
        return k0.copy()
    if t >= 1.0:
        return k1.copy()
    a0 = np.asarray(k0, dtype=np.uint8).astype(np.float64)
    a1 = np.asarray(k1, dtype=np.uint8).astype(np.float64)
    if source_effective is None or background is None:
        out = np.floor(a0 * (1.0 - t) + a1 * t + 0.5).astype(np.uint8)
        return Image.fromarray(out, "RGBA")

    s = np.asarray(source_effective, dtype=np.uint8).astype(np.float64)
    o = np.asarray(background, dtype=np.uint8).astype(np.float64)
    # alpha 全程在 0-255 标度上运算（与前端 JavaScript 的整数标度一致），
    # 避免 /255 再 *255 的往返造成末位差异，确保预览与导出逐位相同。
    alpha0 = a0[..., 3]
    alpha1 = a1[..., 3]
    # 族内判据：K1 的 alpha > 0 说明该像素是真实差异且已解出最小 alpha
    # （K1 alpha = alpha_min * mask，掩码外或微差异置空的像素为 0，不属解族）。
    exact = alpha1 > 0.0
    at255 = alpha0 * (1.0 - t) + alpha1 * t
    # alpha 低于解族下限时反解颜色会越界；钳到下限保证复合精确。
    # 下限取 alpha1 + 0.5：K1 的 alpha 是 8bit 量化值，真实 alpha_min*255
    # 可能比它大半个色阶，抬高半阶可确保反解颜色不越界（免遭裁剪）。
    # 正常像素 alpha0 >= alpha1（源图自身 alpha / 抗锯齿覆盖度即边缘覆盖率），
    # 该钳制只在噪点导致 alpha_min 略大于覆盖率时生效。
    # 上限必须同时钳到 255：否则 floor(255.5 + 0.5) = 256 会在转 uint8 时
    # 回绕成 0，把不透明像素的 alpha 清零（JS 的 Uint8ClampedArray 是饱和
    # 而非回绕，两端都必须显式钳位才可能逐位一致）。
    at255 = np.clip(np.maximum(at255, alpha1 + 0.5), 0.0, 255.0)
    at = at255 / 255.0
    denom = np.where(exact, at, 1.0)
    color = np.clip(o + (s - o) / denom[..., None], 0.0, 255.0)
    family = np.empty_like(a0)
    family[..., :3] = color
    family[..., 3] = at255
    linear = a0 * (1.0 - t) + a1 * t
    out = np.floor(np.where(exact[..., None], family, linear) + 0.5).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def render_at(k0: Image.Image, k1: Image.Image, t: float = 0.0,
              corner_markers: bool = DEFAULT_CORNER_MARKERS,
              source_effective: Image.Image | None = None,
              background: Image.Image | None = None) -> Image.Image:
    """按透明判定 t 渲染最终结果；有 sidecar 时沿精确 alpha 解族渲染。"""
    img = interpolate_keyframes(k0, k1, t, source_effective, background)
    if corner_markers:
        arr = np.array(img)  # 可写副本
        _draw_corner_markers(arr)
        img = Image.fromarray(arr, "RGBA")
    return img


def compute_diff(a_img: Image.Image, b_img: Image.Image, tolerance: int = 0,
                 source: str = "a", workers: int | None = None,
                 min_pixels_for_mp: int = DEFAULT_MIN_PIXELS_FOR_MP,
                 feather_radius: int = DEFAULT_FEATHER_RADIUS,
                 transparency: float = 0.0,
                 corner_markers: bool = DEFAULT_CORNER_MARKERS,
                 despeckle: int = DEFAULT_DESPECKLE,
                 semi_attribution: bool = DEFAULT_SEMI_ATTRIBUTION):
    """
    计算两张图片的像素级差异并渲染，返回 (透明背景输出图, 统计信息)。

    默认（semi_attribution=False）返回**羽化素材图层**：差异像素取来源图原色、
    alpha=255（掩码内部），掩码外侧按 feather_radius 做 alpha 渐变并按精确解族
    反解颜色。因此在原底图上逐像素精确还原来源图，换到新底图时边缘自然过渡。
    feather_radius=0 时等同于纯原色提取（硬边）。

    参数
    ----
    feather_radius : int  边缘羽化半径 0-3（默认 1）；0 = 不羽化（硬边）
    transparency : float 透明判定 0.0-1.0（仅 semi_attribution=True 时有效）。
                  0 = 差异视为内容变化；1 = 差异完全归因于透明度；
                  中间值沿精确 alpha 解族渲染。
    corner_markers : bool 是否在输出四角绘制定位像素（默认开启）
    despeckle   : int  杂色抑制档位 0=关闭 / 1=标准 / 2=强（仅半透明归因路径生效）
    semi_attribution : bool 是否启用「半透明归因」旧版算法（默认 False）。

    返回
    ----
    (PIL.Image.Image, stats: dict)

    另见 compute_outputs()：一次拿到「原色提取图」与「羽化素材图层」两张。
    """
    k0, k1, stats = compute_keyframes(
        a_img, b_img, tolerance=tolerance, source=source, workers=workers,
        min_pixels_for_mp=min_pixels_for_mp, feather_radius=feather_radius,
        despeckle=despeckle, semi_attribution=semi_attribution,
    )
    stats["corner_markers"] = bool(corner_markers)
    if not semi_attribution:
        # 默认路径：直接返回羽化素材图层 K1（feather_radius=0 时即 K0）。
        return render_at(k0, k1, 1.0, corner_markers), stats

    t = min(1.0, max(0.0, float(transparency)))
    stats["transparency"] = round(t, 4)
    # 中间判定值需要 sidecar 才能沿精确 alpha 解族渲染；端点直接返回关键帧
    aux_s = aux_o = None
    if 0.0 < t < 1.0:
        aux_s, aux_o = make_render_auxiliary(a_img, b_img, source)
    return render_at(k0, k1, t, corner_markers,
                     source_effective=aux_s, background=aux_o), stats


def compute_outputs(a_img: Image.Image, b_img: Image.Image, tolerance: int = 0,
                    source: str = "a", workers: int | None = None,
                    feather_radius: int = DEFAULT_FEATHER_RADIUS,
                    corner_markers: bool = False,
                    despeckle: int = DEFAULT_DESPECKLE,
                    semi_attribution: bool = DEFAULT_SEMI_ATTRIBUTION):
    """
    一次得到两张产出（用户双产出需求）：

    plain     : 原色提取图 —— alpha 仅 0/255（硬边），差异像素取来源图原色。
    feathered : 羽化素材图层 —— 掩码外侧 alpha 渐变，可直接放到新背景上。

    feather_radius=0 时两者像素相同。返回 (plain, feathered, stats)。
    """
    k0, k1, stats = compute_keyframes(
        a_img, b_img, tolerance=tolerance, source=source, workers=workers,
        feather_radius=feather_radius, despeckle=despeckle,
        semi_attribution=semi_attribution,
    )
    stats["corner_markers"] = bool(corner_markers)
    plain = render_at(k0, k0, 0.0, corner_markers)
    feathered = render_at(k0, k1, 1.0, corner_markers)
    return plain, feathered, stats


# ---------------------------------------------------------------------------
# 输出与缓存
# ---------------------------------------------------------------------------
def save_png(img: Image.Image, path) -> str:
    """
    把图像保存为 PNG，返回保存后的路径字符串。

    性能说明：不使用 Pillow 的 ``optimize=True``。实测 1920x1080 结果图
    optimize=True 需约 2448 ms、optimize=False 仅约 368 ms（快 6.6 倍），
    而两者解码后的像素完全一致（体积仅相差约 3%）。开关「优化」会让编码器
    对每行滤波器做穷举并在压缩参数间搜索，是整条流水线里最慢的一环，
    远超过差异计算本身（94 ms），因此改回默认的 zlib 默认级别。
    """
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img.save(str(p), "PNG")
    return str(p)


def cleanup_cache(cache_dir, days: float = 7):
    """
    清理缓存目录中修改时间早于 days 天的文件（不递归子目录）。

    返回被删除的文件名列表；目录不存在时返回空列表。
    """
    deleted = []
    try:
        d = Path(cache_dir)
    except Exception:
        return deleted

    if not d.exists() or not d.is_dir():
        return deleted

    try:
        cutoff = time.time() - float(days) * 86400.0
    except Exception:
        return deleted

    try:
        entries = list(d.iterdir())
    except Exception:
        return deleted

    for entry in entries:
        try:
            if not entry.is_file():
                continue
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
                deleted.append(entry.name)
        except Exception as exc:
            print(f"[diff_core] 清理缓存失败：{entry}（{exc}）", file=sys.stderr)

    return deleted


def make_preview_dataurl(img: Image.Image, max_size: int = DEFAULT_PREVIEW_SIZE) -> str:
    """
    生成预览图 data URL（JPEG，便于前端 <img> 直接展示）。

    保持比例缩放到最长边 max_size；若图片含透明像素，
    先贴到深灰纯色底 (#1f2430) 上避免出现黑块。
    """
    im = img.convert("RGBA")
    im.thumbnail((int(max_size), int(max_size)), Image.LANCZOS)

    # 合成到深灰底
    bg = Image.new("RGB", im.size, PREVIEW_BG_COLOR)
    bg.paste(im, (0, 0), im)  # 用 alpha 作为蒙版合成

    buf = io.BytesIO()
    bg.save(buf, "JPEG", quality=82)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + data


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diff_core.py",
        description="图片差异提取工具：比较两张图片，输出差异像素的透明背景 PNG。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python diff_core.py a.png b.png -o diff.png\n"
            "  python diff_core.py a.png b.png --tolerance 5 --source b --workers 8\n"
        ),
    )
    parser.add_argument("image_a", help="提取图片 A 的路径")
    parser.add_argument("image_b", help="背景图片 B 的路径")
    parser.add_argument("-o", "--output", default="./diff_out.png",
                        help="输出 PNG 路径（默认 ./diff_out.png）")
    parser.add_argument("--tolerance", type=int, default=0,
                        help="容差 0-255，任一通道差值大于该值才算差异（默认 0）")
    parser.add_argument("--source", choices=["a", "b"], default="a",
                        help="差异像素的颜色取自哪张图（默认 a）")
    parser.add_argument("--workers", type=int, default=0,
                        help="并行进程数，0 表示自动（默认 0）")
    parser.add_argument("--feather", "--aa-radius", dest="feather", type=int,
                        default=DEFAULT_FEATHER_RADIUS,
                        help=f"边缘羽化半径 0-{MAX_FEATHER_RADIUS}（像素），"
                             f"0=不羽化硬边（默认 {DEFAULT_FEATHER_RADIUS}）。"
                             f"--aa-radius 为已废弃的旧参数名，等价于 --feather")
    parser.add_argument("--plain-output", default="",
                        help="额外输出一张「原色提取图」（不含羽化）到该路径")
    parser.add_argument("--corner-markers", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_CORNER_MARKERS,
                        help="在输出四角绘制定位像素，便于 PS 定位（默认开启；"
                             "--no-corner-markers 关闭）")
    parser.add_argument("--semi-attribution", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_SEMI_ATTRIBUTION,
                        help="启用「半透明归因」旧版算法（默认关闭）：对差异做 un-blending 反解，"
                             "可能产生严重杂色；默认关闭时输出叠加底图逐像素精确还原来源图")
    parser.add_argument("--transparency", type=float, default=DEFAULT_TRANSPARENCY * 100,
                        help="透明判定百分比 0-100（仅 --semi-attribution 时有效，默认 0）")
    parser.add_argument("--despeckle", type=int, default=DEFAULT_DESPECKLE,
                        choices=[0, 1, 2],
                        help="杂色抑制档位：0=关闭 / 1=标准 / 2=强（默认 1）。"
                             "抑制 un-blending 噪点放大形成的彩色杂点")
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS_FOR_MP,
                        help=f"启用多进程的最小总像素数（默认 {DEFAULT_MIN_PIXELS_FOR_MP}）")
    return parser


def main(argv=None) -> int:
    """CLI 主函数，返回进程退出码。"""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        try:
            a_img = decode_source({"type": "path", "value": args.image_a})
        except ValueError as exc:
            raise ValueError(f"图片 A 读取失败：{exc}") from exc

        try:
            b_img = decode_source({"type": "path", "value": args.image_b})
        except ValueError as exc:
            raise ValueError(f"图片 B 读取失败：{exc}") from exc

        workers = args.workers if args.workers and args.workers > 0 else None
        result, stats = compute_diff(
            a_img, b_img,
            tolerance=args.tolerance,
            source=args.source,
            workers=workers,
            min_pixels_for_mp=args.min_pixels,
            feather_radius=args.feather,
            corner_markers=args.corner_markers,
            transparency=args.transparency / 100.0,
            despeckle=args.despeckle,
            semi_attribution=args.semi_attribution,
        )

        out_path = save_png(result, args.output)

        plain_path = ""
        if args.plain_output:
            plain, _f, _s = compute_outputs(
                a_img, b_img,
                tolerance=args.tolerance,
                source=args.source,
                workers=workers,
                feather_radius=args.feather,
                corner_markers=args.corner_markers,
                despeckle=args.despeckle,
                semi_attribution=args.semi_attribution,
            )
            plain_path = save_png(plain, args.plain_output)

        bbox_text = "无" if stats["bbox"] is None else (
            f"[{stats['bbox'][0]}, {stats['bbox'][1]}] - [{stats['bbox'][2]}, {stats['bbox'][3]}]"
        )
        ab = stats["alpha_breakdown"]

        print("=" * 56)
        print("图片差异提取完成")
        print("=" * 56)
        print(f"图片 A 尺寸 : {stats['a_size'][0]} x {stats['a_size'][1]}")
        print(f"图片 B 尺寸 : {stats['b_size'][0]} x {stats['b_size'][1]}")
        print(f"输出尺寸    : {stats['width']} x {stats['height']}")
        print(f"容差        : {stats['tolerance']}")
        print(f"取色来源    : {'A' if stats['source'] == 'a' else 'B'}")
        print(f"边缘羽化    : {stats['feather_radius']} 像素"
              + ("（不羽化，硬边）" if stats['feather_radius'] == 0 else ""))
        print(f"定位像素    : {'开启（四角纯红标记）' if stats['corner_markers'] else '关闭'}")
        if stats.get("semi_attribution"):
            print(f"半透明归因  : 开启（旧版算法，un-blending 边缘可能有杂色）")
            print(f"透明判定    : {stats['transparency'] * 100:.0f}%"
                  + ("（差异视为内容变化）" if stats['transparency'] == 0
                     else "（差异归因于透明度）" if stats['transparency'] >= 1 else "（两端插值）"))
        else:
            print(f"半透明归因  : 关闭（原色提取 + 边缘羽化）")
        print(f"差异像素数  : {stats['diff_pixels']}")
        print(f"总像素数    : {stats['total_pixels']}")
        print(f"差异占比    : {stats['ratio'] * 100:.2f}%  (ratio={stats['ratio']})")
        print(f"差异外接框  : {bbox_text}")
        print(f"透明度构成  : 不透明 {ab['opaque']} / 半透明 {ab['semi']} / 全透明 {ab['transparent']}")
        print(f"计算引擎    : {stats['engine']}")
        print(f"进程数      : {stats['workers']}")
        print(f"耗时        : {stats['elapsed_ms']} ms")
        print(f"输出路径    : {out_path}")
        if plain_path:
            print(f"原色图路径  : {plain_path}")
        print("=" * 56)
        return 0

    except ValueError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[错误] 用户中断", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[错误] 处理失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
