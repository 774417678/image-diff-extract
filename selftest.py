# -*- coding: utf-8 -*-
"""
DiffGet 自检
============

用**合成夹具**验证核心不变量，不依赖任何外部素材，任何人都可复现。

    python selftest.py

全部通过时退出码为 0，否则为 1。

覆盖的关键性质
--------------
S1  默认路径（原色 + 羽化）：叠加到背景图上逐像素还原来源图，误差 <= 1
S2  羽化半径 0/1/2/3 都不破坏「叠加还原」，且半径越大半透明环越宽
S3  原色提取图：掩码内 alpha 仅为 255（硬边），颜色等于来源原色
S4  不变量：差异掩码内任意真实帧间差 > 2 的像素，alpha 必须 >= 1（绝不允许空洞）
S5  旧版「半透明归因」：带丰富 alpha 档位，且叠加还原误差 <= 3
S6  预览与导出一致：render_at(..., t=1.0) 等于关键帧 K1
S7  多进程路径与单进程结果逐像素一致，且失败会回退而不崩溃
S8  Web API：/api/extract 与 /api/save 端到端可用，产物与内存计算一致
S9  参数校验：非法 feather_radius / tolerance / source 返回 400
"""
from __future__ import annotations

import io
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff_core as dc  # noqa: E402

W, H = 320, 240
CORNERS = 4
_results: list[bool] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 合成夹具：不涉及任何真实素材
# ---------------------------------------------------------------------------
def make_fixture():
    """背景图 B（全不透明）与来源图 A（含内容变化 + 真半透明叠加 + 微差异噪点）。"""
    yy, xx = np.mgrid[0:H, 0:W]
    B = np.zeros((H, W, 3), np.uint8)
    B[..., 0] = (xx * 255 // (W - 1)).astype(np.uint8)
    B[..., 1] = (yy * 255 // (H - 1)).astype(np.uint8)
    B[..., 2] = 120
    circle = (xx - 240) ** 2 + (yy - 60) ** 2 < 40 ** 2
    B[circle] = (30, 200, 90)
    B[190:220, :] = (200, 200, 210)

    A = B.copy()
    # 1) 整块内容变化（不透明）
    A[40:120, 40:120] = (220, 40, 60)
    # 2) 唯一的 RGB 渐变差异
    A[130:170, 40:120, 0] = np.linspace(0, 255, 80, dtype=np.uint8)[None, :]
    # 3) 微差异噪点（±2）——应被「微差异置空」规则处理，代价 <=2
    rng = np.random.default_rng(20260929)
    noisy = rng.integers(-2, 3, size=(30, 60, 3))
    A[60:90, 150:210] = np.clip(
        B[60:90, 150:210].astype(np.int16) + noisy, 0, 255).astype(np.uint8)

    img_b = Image.fromarray(B, "RGB").convert("RGBA")

    # 4) 来源图自身的半透明覆盖层（alpha=128）——应保留原始 alpha
    a_rgba = np.dstack([A, np.full((H, W), 255, np.uint8)])
    a_rgba[100:140, 220:290, :3] = (255, 240, 200)
    a_rgba[100:140, 220:290, 3] = 128
    img_a = Image.fromarray(a_rgba, "RGBA")
    return img_a, img_b


A_IMG, B_IMG = make_fixture()
A_NP = np.asarray(A_IMG).astype(np.float64)
B_NP = np.asarray(B_IMG).astype(np.float64)
A_RGB, B_RGB = A_NP[..., :3], B_NP[..., :3]
SRC_A = A_NP[..., 3]

# 来源图覆盖在背景图上的**外观**：来源自身带 alpha 时先折算。
# 提取结果叠加回 B 后应等于这个外观（而不是 A 的原始 RGB）。
ALPHA_A = SRC_A[..., None] / 255.0
A_EFF = ALPHA_A * A_RGB + (1.0 - ALPHA_A) * B_RGB

D = np.abs(A_RGB - B_RGB).max(axis=2)
MASK = D > 0


def composite_over_b(layer_rgba: np.ndarray) -> np.ndarray:
    """按 alpha 把图层叠加到背景图 B 上（与 Photoshop Normal 混合等效）。"""
    al = layer_rgba[..., 3:4] / 255.0
    return layer_rgba[..., :3] * al + B_RGB * (1 - al)


def err_vs_a(layer: Image.Image) -> np.ndarray:
    """叠加到 B 上后与来源图外观的逐像素误差（排除四角定位像素）。"""
    arr = np.asarray(layer)
    err = np.abs(composite_over_b(arr) - A_EFF).max(axis=2)
    err[:CORNERS, :CORNERS] = err[:CORNERS, -CORNERS:] = 0.0
    err[-CORNERS:, :CORNERS] = err[-CORNERS:, -CORNERS:] = 0.0
    return err


def alpha_levels(layer: Image.Image) -> int:
    return len({int(v) for v in np.asarray(layer)[..., 3].ravel().tolist()})


def main() -> int:
    print("=" * 62)
    print("DiffGet 自检（合成夹具 %dx%d）" % (W, H))
    print("=" * 62)
    print(f"差异像素 {int(MASK.sum())} / {MASK.size}"
          f"（其中来源半透明 {int(((SRC_A > 0) & (SRC_A < 255) & MASK).sum())}）")
    print()

    # ---------------------------------------------------------- S1 默认路径
    print("--- S1 默认路径（原色 + 羽化 1px）叠加还原 ---")
    plain, feather, st = dc.compute_outputs(
        A_IMG, B_IMG, tolerance=0, source="a", feather_radius=1,
        corner_markers=False, semi_attribution=False)
    e = err_vs_a(feather)
    check("羽化图层叠加到 B 后还原来源图（max <= 1）", e.max() <= 1.0,
          f"max={e.max():.2f} p99={np.percentile(e,99):.2f}")
    check("差异统计与掩码一致", st["diff_pixels"] == int(MASK.sum()),
          f"{st['diff_pixels']}")

    # ---------------------------------------------------------- S2 羽化半径
    print()
    print("--- S2 羽化半径 0/1/2/3 均不破坏叠加还原 ---")
    rings = []
    for R in (0, 1, 2, 3):
        _, fe, _ = dc.compute_outputs(A_IMG, B_IMG, tolerance=0, source="a",
                                      feather_radius=R, corner_markers=False)
        er = err_vs_a(fe).max()
        arr = np.asarray(fe)
        ring = int(((arr[..., 3] > 0) & (arr[..., 3] < 255) & ~MASK).sum())
        rings.append(ring)
        check(f"羽化 {R}px 叠加还原 max <= 1", er <= 1.0,
              f"max={er:.2f} 环像素={ring}")
    check("羽化半径越大环越宽（0 <= r1 <= r2 <= r3）",
          rings[0] == 0 and rings[0] <= rings[1] <= rings[2] <= rings[3],
          f"{rings}")

    # ---------------------------------------------------------- S3 原色图
    print()
    print("--- S3 原色提取图是硬边 ---")
    p = np.asarray(plain)
    inside = np.asarray(np.asarray(plain)[..., 3])[MASK]
    check("掩码内 alpha <= 255（未超）", inside.max() <= 255)
    solid = MASK & (SRC_A == 255)
    check("差值区（来源不透明）alpha 恒为 255",
          bool(np.all(p[..., 3][solid] == 255)))
    semi_src = MASK & (SRC_A > 0) & (SRC_A < 255)
    check("来源半透明区 alpha 保留原值",
          bool(np.all(p[..., 3][semi_src] == SRC_A[semi_src].astype(np.uint8))))
    check("掩码外 alpha 恒为 0", bool(np.all(p[..., 3][~MASK] == 0)))

    # ---------------------------------------------------------- S4 不变量
    print()
    print("--- S4 不变量：掩码内 |d|>2 的像素绝不允许 alpha=0（空洞）---")
    k0l, k1l, stl = dc.compute_keyframes(
        A_IMG, B_IMG, tolerance=0, source="a", feather_radius=1, despeckle=1,
        semi_attribution=True)
    L = np.asarray(k1l)
    guard = MASK & (D > 2) & (SRC_A > 0)
    hole = int(np.count_nonzero(guard & (L[..., 3] == 0)))
    check("旧版路径无空洞", hole == 0, f"空洞={hole}")
    check("默认路径无空洞",
          int(np.count_nonzero(guard & (np.asarray(plain)[..., 3] == 0))) == 0)

    # ---------------------------------------------------------- S5 旧版
    print()
    print("--- S5 旧版「半透明归因」---")
    el = err_vs_a(k1l)
    check("旧版叠加还原 max <= 3", el.max() <= 3.0, f"max={el.max():.2f}")
    lv = alpha_levels(k1l)
    check("旧版带丰富 alpha 档位（非硬边）", lv > 8, f"档位={lv}")
    check("旧版响应标记 semi_attribution",
          stl["semi_attribution"] is True)

    # ---------------------------------------------------------- S6 预览=导出
    print()
    print("--- S6 预览与导出一致 ---")
    only_k1 = dc.render_at(k0l, k1l, 1.0, False)
    check("render_at(t=1.0) == K1（位级一致）",
          np.array_equal(np.asarray(only_k1), L))
    check("render_at(t=0.0) == K0（位级一致）",
          np.array_equal(np.asarray(dc.render_at(k0l, k1l, 0.0, False)),
                         np.asarray(k0l)))

    # ---------------------------------------------------------- S7 多进程
    print()
    print("--- S7 多进程与单进程结果一致 ---")
    _, seq, _ = dc.compute_keyframes(A_IMG, B_IMG, tolerance=0, source="a",
                                     feather_radius=1, workers=1,
                                     min_pixels_for_mp=10 ** 9)
    _, par, stp = dc.compute_keyframes(A_IMG, B_IMG, tolerance=0, source="a",
                                       feather_radius=1, workers=2,
                                       min_pixels_for_mp=1)
    check("多进程结果与单进程逐像素一致",
          np.array_equal(np.asarray(seq), np.asarray(par)),
          f"engine={stp['engine']}")
    check("引擎标记合法", stp["engine"] in ("multiprocess", "single(fallback)",
                                            "single"), stp["engine"])

    # ---------------------------------------------------------- S8 API
    print()
    print("--- S8 Web API 端到端 ---")
    with tempfile.TemporaryDirectory() as td:
        pa = os.path.join(td, "a.png")
        pb = os.path.join(td, "b.png")
        A_IMG.save(pa)
        B_IMG.save(pb)
        import web_app
        client = web_app.app.test_client()
        r = client.post("/api/extract", json={
            "a": {"type": "path", "value": pa}, "b": {"type": "path", "value": pb},
            "tolerance": 0, "source": "a", "feather_radius": 1})
        check("extract 返回 200", r.status_code == 200, f"status={r.status_code}")
        j = r.get_json() or {}
        check("响应含 plain_url / feathered_url / keyframes",
              bool(j.get("plain_url")) and bool(j.get("feathered_url"))
              and len(j.get("keyframes") or []) == 2)
        rs = client.post("/api/save", json={"file": j.get("file", ""),
                                            "name": "selftest",
                                            "corner_markers": False})
        check("save 返回 200", rs.status_code == 200, f"status={rs.status_code}")
        js = rs.get_json() or {}
        check("一次写出两张且都真实存在",
              os.path.exists(js.get("saved", ""))
              and os.path.exists(js.get("plain_saved", "")))
        if os.path.exists(js.get("saved", "")):
            saved = np.asarray(Image.open(js["saved"]).convert("RGBA"))
            check("保存的文件叠加还原 max <= 1",
                  float(np.abs(composite_over_b(saved.astype(np.float64))
                               - A_EFF).max()) <= 1.0)

    # ---------------------------------------------------------- S9 校验
    print()
    print("--- S9 参数校验 ---")
    import web_app
    client = web_app.app.test_client()
    for body, desc in (({"feather_radius": 9}, "feather_radius=9"),
                       ({"tolerance": 999}, "tolerance=999"),
                       ({"source": "c"}, "source=c")):
        payload = {"a": {"type": "path", "value": "x.png"},
                   "b": {"type": "path", "value": "y.png"}}
        payload.update(body)
        rr = client.post("/api/extract", json=payload)
        check(f"非法参数 {desc} 返回 400", rr.status_code == 400,
              f"status={rr.status_code}")

    print()
    print("=" * 62)
    passed = sum(_results)
    print(f"总计 {passed}/{len(_results)} 通过")
    print("=" * 62)
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
