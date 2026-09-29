# -*- coding: utf-8 -*-
"""
DiffGet Web 服务
===============

提供 http://127.0.0.1:9426 网页界面，用于将两张图片的差异部分提取为透明 PNG。

路由总览
--------
GET  /                 页面（static/index.html）
GET  /api/info         服务信息（端口、目录、缓存天数、CPU 核数等）
POST /api/load         校验本地图片路径并返回预览图
POST /api/extract      执行差异提取，结果存入缓存目录
GET  /cache/<文件>     读取缓存中的结果图片
POST /api/save         把缓存结果永久保存到 out 目录
POST /api/open_out     打开 out 目录

环境变量
--------
DIFFGET_PORT       服务端口（默认 9426）
DIFFGET_HOST       监听地址（默认 127.0.0.1，设 0.0.0.0 允许局域网访问）
DIFFGET_CACHE_DAYS 缓存保留天数（默认 7）
"""

from __future__ import annotations

import os
import sys
import uuid
import platform
import threading
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

import diff_core

# ---------------------------------------------------------------------------
# 路径与配置
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"   # 结果缓存目录（7 天自动清理）
OUT_DIR = BASE_DIR / "out"       # 永久保存目录

PORT = int(os.environ.get("DIFFGET_PORT", "9426"))
HOST = os.environ.get("DIFFGET_HOST", "127.0.0.1")
CACHE_DAYS = float(os.environ.get("DIFFGET_CACHE_DAYS", "7"))

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")
# data URL 形式的图片可能较大，放宽请求体上限
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024

# 差异提取是 CPU 密集型任务，加锁串行化，避免多个大图任务互相抢核
_EXTRACT_LOCK = threading.Lock()

# Windows 文件名非法字符
_ILLEGAL_NAME_CHARS = '<>:"/\\|?*'


def _err(message: str, status: int):
    """统一错误响应。"""
    return jsonify({"ok": False, "error": message}), status


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    """返回单页应用。"""
    return send_from_directory(app.static_folder or "static", "index.html")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/info")
def api_info():
    """服务信息，用于页面状态展示。"""
    try:
        cache_files = len([p for p in CACHE_DIR.iterdir() if p.is_file()])
    except Exception:
        cache_files = 0
    return jsonify({
        "ok": True,
        "port": PORT,
        "host": HOST,
        "cache_dir": str(CACHE_DIR),
        "out_dir": str(OUT_DIR),
        "cache_days": CACHE_DAYS,
        "cpu_count": os.cpu_count() or 1,
        "cache_files": cache_files,
        "python": platform.python_version(),
    })


@app.post("/api/load")
def api_load():
    """校验本地路径并返回预览图（页面手动输入路径时使用）。"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _err("请求体必须是 JSON 对象", 400)

    path = data.get("path")
    if not isinstance(path, str) or not path.strip():
        return _err("缺少参数 path（图片路径不能为空）", 400)

    try:
        img = diff_core.decode_source({"type": "path", "value": path.strip()})
    except ValueError as exc:
        return _err(str(exc), 400)
    except Exception as exc:
        return _err(f"读取失败：{exc}", 500)

    return jsonify({
        "ok": True,
        "name": Path(path.strip()).name,
        "width": img.width,
        "height": img.height,
        "preview": diff_core.make_preview_dataurl(img),
    })


@app.post("/api/extract")
def api_extract():
    """执行像素级差异提取，结果写入缓存目录。"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _err("请求体必须是 JSON 对象", 400)

    a_src = data.get("a")
    b_src = data.get("b")
    if not isinstance(a_src, dict) or not isinstance(b_src, dict):
        return _err("参数 a 与 b 必须是包含 type 和 value 的对象", 400)

    # 容差、取色来源与边缘羽化参数校验（透明判定在保存时生效，提取只算关键帧）
    tolerance = data.get("tolerance", 0)
    source = data.get("source", "a")
    feather_radius = data.get("feather_radius", diff_core.DEFAULT_FEATHER_RADIUS)
    despeckle = data.get("despeckle", diff_core.DEFAULT_DESPECKLE)
    semi_attr = data.get("semi_attribution", diff_core.DEFAULT_SEMI_ATTRIBUTION)
    if not isinstance(semi_attr, bool):
        semi_attr = str(semi_attr).strip().lower() in ("1", "true", "yes", "on")
    try:
        tolerance = int(tolerance)
    except Exception:
        return _err("容差必须是 0-255 的整数", 400)
    if not (0 <= tolerance <= 255):
        return _err("容差必须在 0-255 之间", 400)
    if str(source).strip().lower() not in ("a", "b"):
        return _err("取色来源只能是 a（提取图片）或 b（背景图片）", 400)
    try:
        feather_radius = int(feather_radius)
    except Exception:
        return _err("羽化半径必须是 0-3 的整数", 400)
    if not (0 <= feather_radius <= diff_core.MAX_FEATHER_RADIUS):
        return _err(f"羽化半径必须在 0-{diff_core.MAX_FEATHER_RADIUS} 之间", 400)
    try:
        despeckle = int(despeckle)
    except Exception:
        return _err("杂色抑制档位必须是 0/1/2 的整数", 400)
    if despeckle not in (0, 1, 2):
        return _err("杂色抑制档位必须是 0（关闭）/ 1（标准）/ 2（强）", 400)

    # 解码两张图片（IO 操作放在锁外）
    try:
        a_img = diff_core.decode_source(a_src)
    except ValueError as exc:
        return _err(f"提取图片读取失败：{exc}", 400)
    except Exception as exc:
        return _err(f"提取图片读取失败：{exc}", 500)
    try:
        b_img = diff_core.decode_source(b_src)
    except ValueError as exc:
        return _err(f"背景图片读取失败：{exc}", 400)
    except Exception as exc:
        return _err(f"背景图片读取失败：{exc}", 500)

    with _EXTRACT_LOCK:
        try:
            k0, k1, stats = diff_core.compute_keyframes(
                a_img, b_img, tolerance=tolerance, source=source,
                feather_radius=feather_radius, despeckle=despeckle,
                semi_attribution=semi_attr,
            )
        except ValueError as exc:
            return _err(f"计算失败：{exc}", 400)
        except Exception as exc:
            return _err(f"计算失败：{exc}", 500)

        # 双产出：k0 = 原色提取图（硬边），k1 = 羽化素材图层。
        # 羽化半径 0 时两者像素相同，但都缓存以便「保存」一次写出两张。
        base = f"diff_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        k0_name, k1_name = f"{base}_k0.png", f"{base}_k1.png"
        s_name, o_name = f"{base}_s.png", f"{base}_o.png"
        try:
            diff_core.save_png(k0, CACHE_DIR / k0_name)
            diff_core.save_png(k1, CACHE_DIR / k1_name)
            if semi_attr:
                # 仅「半透明归因」旧版路径需要 sidecar（精确 alpha 解族的 S 与 O），
                # 默认路径不生成，省掉两次整幅 PNG 编码
                s_aux, o_aux = diff_core.make_render_auxiliary(a_img, b_img, source)
                diff_core.save_png(s_aux, CACHE_DIR / s_name)
                diff_core.save_png(o_aux, CACHE_DIR / o_name)
        except Exception as exc:
            return _err(f"保存结果失败：{exc}", 500)

        # 每次提取后顺带清理超期缓存
        diff_core.cleanup_cache(CACHE_DIR, CACHE_DAYS)

    return jsonify({
        "ok": True,
        "file": base,
        "url": f"/cache/{k0_name}",
        "plain_url": f"/cache/{k0_name}",
        "feathered_url": f"/cache/{k1_name}",
        "keyframes": [f"/cache/{k0_name}", f"/cache/{k1_name}"],
        # 默认路径不生成 sidecar，前端因此走单幅贴图预览（无插值/无反解）
        "render_aux": ([f"/cache/{s_name}", f"/cache/{o_name}"] if semi_attr else None),
        "width": stats["width"],
        "height": stats["height"],
        "diff_pixels": stats["diff_pixels"],
        "total_pixels": stats["total_pixels"],
        "ratio": stats["ratio"],
        "bbox": stats["bbox"],
        "elapsed_ms": stats["elapsed_ms"],
        "engine": stats["engine"],
        "workers": stats["workers"],
        "size_bytes": (CACHE_DIR / k0_name).stat().st_size,
        "source": stats["source"],
        "tolerance": stats["tolerance"],
        "feather_radius": stats.get("feather_radius", feather_radius),
        "despeckle": stats["despeckle"],
        "semi_attribution": bool(stats.get("semi_attribution", False)),
        "transparency": (None if stats.get("transparency") is None
                         else round(stats["transparency"] * 100)),
        "corner_markers": diff_core.DEFAULT_CORNER_MARKERS,
        "alpha_breakdown": stats["alpha_breakdown"],
        "a_size": stats["a_size"],
        "b_size": stats["b_size"],
        "padded": stats["padded"],
    })


@app.get("/cache/<name>")
def cache_file(name: str):
    """读取缓存目录中的结果图片（带路径穿越防护）。"""
    if not name or "/" in name or "\\" in name or ".." in name:
        return _err("非法的文件名", 400)
    if not (CACHE_DIR / name).is_file():
        return _err("文件不存在或已过期被清理", 404)
    return send_from_directory(CACHE_DIR, name, mimetype="image/png")


def _sanitize_save_name(name) -> str:
    """净化用户输入的文件名：只保留 basename、剔除非法字符、确保 .png 后缀。"""
    if not isinstance(name, str) or not name.strip():
        return f"diff_{datetime.now():%Y%m%d_%H%M%S}.png"
    cleaned = Path(name.strip()).name                      # 只取文件名部分
    cleaned = "".join("_" if ch in _ILLEGAL_NAME_CHARS else ch for ch in cleaned)
    cleaned = cleaned.strip().strip(".")                   # 去掉首尾空白和点
    if not cleaned:
        return f"diff_{datetime.now():%Y%m%d_%H%M%S}.png"
    if not cleaned.lower().endswith(".png"):
        cleaned += ".png"
    return cleaned


@app.post("/api/save")
def api_save():
    """把缓存中的结果永久复制到 out 目录。"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _err("请求体必须是 JSON 对象", 400)

    file = data.get("file")
    if not isinstance(file, str) or not file.strip():
        return _err("缺少参数 file（提取结果标识）", 400)
    if "/" in file or "\\" in file or ".." in file:
        return _err("非法的提取结果标识", 400)
    if "_k0.png" in file or "_k1.png" in file:
        return _err("file 应为提取结果标识（不含 _k0/_k1 后缀）", 400)

    # 半透明归因、透明判定与定位像素参数
    t_percent = data.get("t", diff_core.DEFAULT_TRANSPARENCY * 100)
    corner_markers = data.get("corner_markers", diff_core.DEFAULT_CORNER_MARKERS)
    semi_attr = data.get("semi_attribution", None)   # None = 沿用提取时的设置
    if semi_attr is not None and not isinstance(semi_attr, bool):
        semi_attr = str(semi_attr).strip().lower() in ("1", "true", "yes", "on")
    try:
        t_percent = float(t_percent)
    except Exception:
        return _err("透明判定必须是 0-100 的数字", 400)
    if not (0 <= t_percent <= 100):
        return _err("透明判定必须在 0-100 之间", 400)
    if not isinstance(corner_markers, bool):
        corner_markers = str(corner_markers).strip().lower() in ("1", "true", "yes", "on")

    # 载入两个关键帧：k0 = 原色提取图（硬边），k1 = 羽化素材图层
    k0_path = CACHE_DIR / f"{file}_k0.png"
    k1_path = CACHE_DIR / f"{file}_k1.png"
    s_path = CACHE_DIR / f"{file}_s.png"
    o_path = CACHE_DIR / f"{file}_o.png"
    if not k0_path.is_file() or not k1_path.is_file():
        return _err(f"提取结果不存在（可能已被清理）：{file}", 400)
    try:
        with Image.open(k0_path) as im:
            k0 = im.convert("RGBA")
        with Image.open(k1_path) as im:
            k1 = im.convert("RGBA")
        s_aux = o_aux = None
        if s_path.is_file() and o_path.is_file():
            with Image.open(s_path) as im:
                s_aux = im.convert("RGB")
            with Image.open(o_path) as im:
                o_aux = im.convert("RGB")
        # 默认路径：plain = K0（原色、硬边）、feathered = K1（边界羽化）。
        # 只有半透明归因（且缓存中存在 sidecar）时才走精确 alpha 解族插值。
        # 注意：未显式传 semi_attribution 时按全局默认（关闭）处理，
        # 否则会误走旧版 un-blending 分支、把「羽化图」保存成原色图。
        if semi_attr is None:
            semi_attr = diff_core.DEFAULT_SEMI_ATTRIBUTION
        use_aux = bool(semi_attr)
        plain = diff_core.render_at(k0, k0, 0.0, corner_markers)
        if use_aux:
            feathered = diff_core.render_at(
                k0, k1, t_percent / 100.0, corner_markers,
                source_effective=s_aux, background=o_aux,
            )
        else:
            feathered = diff_core.render_at(k0, k1, 1.0, corner_markers)
    except Exception as exc:
        return _err(f"渲染失败：{exc}", 500)

    # 一次写出两张：<名字>.png（羽化素材）与 <名字>_原始.png（原色提取）
    dest = OUT_DIR / _sanitize_save_name(data.get("name"))
    if dest.exists():  # 重名自动加序号
        stem = dest.stem
        idx = 1
        while dest.exists():
            dest = OUT_DIR / f"{stem}_{idx}.png"
            idx += 1
    plain_dest = dest.with_name(f"{dest.stem}_原始.png")
    if plain_dest.exists():
        idx = 1
        while plain_dest.exists():
            plain_dest = dest.with_name(f"{dest.stem}_原始_{idx}.png")
            idx += 1

    try:
        diff_core.save_png(feathered, dest)
        diff_core.save_png(plain, plain_dest)
    except Exception as exc:
        return _err(f"保存失败：{exc}", 500)

    return jsonify({
        "ok": True,
        "saved": str(dest.resolve()),
        "name": dest.name,
        "plain_saved": str(plain_dest.resolve()),
        "plain_name": plain_dest.name,
        "size_bytes": dest.stat().st_size,
        "t": round(t_percent),
        "semi_attribution": bool(semi_attr) if semi_attr is not None else None,
    })


@app.post("/api/open_out")
def api_open_out():
    """在文件管理器中打开 out 目录。"""
    try:
        if os.name == "nt":
            os.startfile(str(OUT_DIR))  # noqa: S606  (仅 Windows 分支调用)
        else:
            subprocess.Popen(["xdg-open", str(OUT_DIR)])
    except Exception as exc:
        return _err(f"无法打开目录 {OUT_DIR}：{exc}", 500)
    return jsonify({"ok": True, "path": str(OUT_DIR)})


# ---------------------------------------------------------------------------
# 错误处理（统一返回 JSON）
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def _handle_404(_exc):
    return _err("接口不存在", 404)


@app.errorhandler(405)
def _handle_405(_exc):
    return _err("方法不允许", 405)


@app.errorhandler(413)
def _handle_413(_exc):
    return _err("图片数据过大（上限 256MB），请压缩后重试", 413)


@app.errorhandler(500)
def _handle_500(_exc):
    return _err("服务器内部错误", 500)


@app.errorhandler(Exception)
def _handle_unhandled(exc):
    """兜底：保证任何异常都以 JSON 形式返回。"""
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return _err(exc.description or "请求失败", exc.code or 500)
    return _err(f"服务器内部错误：{exc}", 500)


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def _open_browser() -> None:
    """延迟打开浏览器（等服务先就绪）。"""
    try:
        webbrowser.open(f"http://127.0.0.1:{PORT}")
    except Exception:
        pass


def main() -> None:
    # 启动时清理超期缓存
    deleted = diff_core.cleanup_cache(CACHE_DIR, CACHE_DAYS)
    if deleted:
        print(f"[缓存清理] 已删除 {len(deleted)} 个超过 {CACHE_DAYS:g} 天的缓存文件")

    print("=" * 60)
    print("  DiffGet · 图片差异提取工具")
    print("=" * 60)
    print(f"  服务地址    : http://127.0.0.1:{PORT}")
    print(f"  监听地址    : {HOST}")
    print(f"  缓存目录    : {CACHE_DIR}（保留 {CACHE_DAYS:g} 天）")
    print(f"  保存目录    : {OUT_DIR}")
    print(f"  CPU 核心数  : {os.cpu_count() or 1}")
    print(f"  Python      : {platform.python_version()}")
    print("  浏览器将自动打开；关闭本窗口即停止服务")
    print("=" * 60)

    threading.Timer(1.2, _open_browser).start()

    try:
        app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)
    except OSError as exc:
        print(f"[错误] 服务启动失败（端口 {PORT} 可能被占用）：{exc}", file=sys.stderr)
        print("       可更换端口后重试，例如：set DIFFGET_PORT=9427 && python web_app.py",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
