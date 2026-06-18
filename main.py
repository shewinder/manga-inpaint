"""
manga-inpaint: 漫画文字检测 + 擦除 + 嵌字服务

POST /detect            CRAFT 检测 → bbox 列表
POST /inpaint_and_render  擦除 + 嵌字 → 结果图片
"""
import io
import os
import json
import base64
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

# ---- render 模块 ----
from render import render as do_render
from render import Bbox as RenderBbox
from render import Translation as RenderTranslation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manga-inpaint")

app = FastAPI(title="manga-inpaint", version="2.0.0")

# ---- numpy 兼容性修补 ----
_original_array = np.array

def _compat_array(obj, *args, **kwargs):
    if 'dtype' not in kwargs:
        try:
            return _original_array(obj, *args, **kwargs)
        except (ValueError, TypeError):
            kwargs = {**kwargs, 'dtype': object}
            return _original_array(obj, *args, **kwargs)
    return _original_array(obj, *args, **kwargs)

# ---- 模型延迟加载 ----
craft_net = None
refine_net = None
lama_model = None
manga_ocr = None

# ---- 配置 ----
CONFIDENCE_THRESHOLD = 0.8
DEFAULT_LINK_THRESHOLD = 0.25
app.state.link_threshold = DEFAULT_LINK_THRESHOLD
MASK_DILATION = 6
INPAINT_MAX_SIZE = 2048
BASE_DIR = Path(__file__).parent
FONT_DIR = BASE_DIR / "fonts"
MODEL_DIR = BASE_DIR / "models"

# 项目本地模型路径
CRAFT_WEIGHT_PATH = str(MODEL_DIR / "craft" / "craft_mlt_25k.pth")
REFINER_WEIGHT_PATH = str(MODEL_DIR / "craft" / "craft_refiner_CTW1500.pth")
LAMA_MODEL_PATH = str(MODEL_DIR / "lama" / "big-lama.pt")
MANGA_OCR_PATH = str(MODEL_DIR / "manga-ocr-flat")


# ---- 会话数据缓存 ----
# 用 data_id 索引，避免 AI 手动管理 bbox 文件
import uuid
_session_store: dict = {}  # {data_id: {bboxes, results, image_path, ...}}


def _save_session(**kwargs) -> str:
    data_id = uuid.uuid4().hex[:12]
    _session_store[data_id] = kwargs
    return data_id


def _get_session(data_id: str) -> dict:
    if data_id not in _session_store:
        raise HTTPException(404, f"data_id 不存在或已过期: {data_id}")
    return _session_store[data_id]


# ---- 数据模型 ----
class BboxOut(BaseModel):
    id: int
    x: int
    y: int
    w: int
    h: int
    polygon: List[List[float]]


class DetectResponse(BaseModel):
    bboxes: List[BboxOut]


class TranslationIn(BaseModel):
    bbox_id: int = 0
    text: str
    color: str = "#000000"
    direction: str = "auto"
    bbox_ids: Optional[List[int]] = None


class InpaintAndRenderRequest(BaseModel):
    translations: List[TranslationIn]
    font_dir: str = ""


class InpaintAndRenderResponse(BaseModel):
    rendered_image: str  # base64


# ---- 核心函数 ----

def _ensure_file(url: str, dst: str, desc: str = ""):
    if os.path.exists(dst):
        return
    logger.info(f"下载 {desc}: {dst}")
    import httpx
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    logger.info(f"{desc} 下载完成")


def load_craft():
    global craft_net, refine_net
    if craft_net is None:
        import torchvision.models.vgg as vgg
        if not hasattr(vgg, 'model_urls'):
            vgg.model_urls = {
                "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            }

        _ensure_file(
            "https://hf-mirror.com/Manbehindthemadness/craft_mlt_25k/resolve/main/craft_mlt_25k.pth",
            CRAFT_WEIGHT_PATH, "CRAFT检测")
        _ensure_file(
            "https://hf-mirror.com/Manbehindthemadness/craft_mlt_25k/resolve/main/craft_refiner_CTW1500.pth",
            REFINER_WEIGHT_PATH, "CRAFT精炼")

        np.array = _compat_array
        try:
            from craft_text_detector import load_craftnet_model, load_refinenet_model
            logger.info("加载 CRAFT 模型...")
            craft_net = load_craftnet_model(cuda=True, weight_path=CRAFT_WEIGHT_PATH)
            refine_net = load_refinenet_model(cuda=True, weight_path=REFINER_WEIGHT_PATH)
            logger.info("CRAFT 模型加载完成")
        finally:
            np.array = _original_array


def load_lama():
    global lama_model
    if lama_model is None:
        _ensure_file(
            "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
            LAMA_MODEL_PATH, "LaMa擦除")
        logger.info("加载 LaMa 模型...")
        os.environ["LAMA_MODEL"] = LAMA_MODEL_PATH
        from simple_lama_inpainting import SimpleLama
        lama_model = SimpleLama()
        logger.info("LaMa 模型加载完成")


def load_ocr():
    global manga_ocr
    if manga_ocr is None:
        if not os.path.exists(MANGA_OCR_PATH):
            logger.info(f"下载 manga-ocr 模型到 {MANGA_OCR_PATH}...")
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from huggingface_hub import snapshot_download
            snapshot_download("kha-white/manga-ocr-base", local_dir=MANGA_OCR_PATH)
            logger.info("manga-ocr 模型下载完成")
        logger.info("加载 manga-ocr 模型...")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from manga_ocr import MangaOcr
        manga_ocr = MangaOcr(pretrained_model_name_or_path=MANGA_OCR_PATH)
        logger.info("manga-ocr 模型加载完成")


def create_mask(image_shape: tuple, bboxes: list, dilation: int = MASK_DILATION) -> np.ndarray:
    import cv2
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for box in bboxes:
        polygon = np.array(box["polygon"], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def detect_text(image: np.ndarray, link_threshold: float = 0) -> List[dict]:
    from craft_text_detector import get_prediction, empty_cuda_cache

    load_craft()

    lt = link_threshold if link_threshold > 0 else app.state.link_threshold
    logger.info(f"开始文字检测, 图片尺寸: {image.shape}, link_threshold={lt}")

    np.array = _compat_array
    try:
        prediction = get_prediction(
            image=image,
            craft_net=craft_net,
            refine_net=refine_net,
            text_threshold=CONFIDENCE_THRESHOLD,
            link_threshold=lt,
            low_text=0.4,
            cuda=True,
            long_size=1280,
        )
    finally:
        np.array = _original_array

    bboxes = []
    for i, box in enumerate(prediction["boxes"]):
        polygon = box.tolist() if hasattr(box, 'tolist') else box
        x = min(p[0] for p in polygon)
        y = min(p[1] for p in polygon)
        w = max(p[0] for p in polygon) - x
        h = max(p[1] for p in polygon) - y
        bboxes.append({
            "id": i,
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "polygon": polygon if isinstance(polygon, list) else polygon.tolist(),
        })

    logger.info(f"检测到 {len(bboxes)} 个文字区域")
    return bboxes


def inpaint(image: np.ndarray, bboxes: list) -> np.ndarray:
    load_lama()
    import torch

    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > INPAINT_MAX_SIZE:
        scale = INPAINT_MAX_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image_small = np.array(Image.fromarray(image).resize((new_w, new_h), Image.LANCZOS))
        scaled_bboxes = []
        for b in bboxes:
            sb = b.copy()
            sb["polygon"] = [[p[0] * scale, p[1] * scale] for p in b["polygon"]]
            scaled_bboxes.append(sb)
        bboxes = scaled_bboxes
    else:
        image_small = image

    mask = create_mask(image_small.shape, bboxes)
    logger.info(f"LaMa 擦除: {image_small.shape[1]}x{image_small.shape[0]}, 掩码 {np.count_nonzero(mask)} px")

    result_small = lama_model(image_small, mask)
    result_small = np.array(result_small)

    if scale != 1.0:
        result_small = np.array(Image.fromarray(result_small).resize((w, h), Image.LANCZOS))

    torch.cuda.empty_cache()
    return result_small


# ---- API 端点 ----

@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    link_threshold: str = Form(""),
):
    """CRAFT 文字检测 — 只检测，不擦除"""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "仅支持图片文件")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"无法读取图片: {e}")

    lt = float(link_threshold) if link_threshold.strip() else 0
    img_array = np.array(image)
    bboxes = detect_text(img_array, link_threshold=lt)
    return DetectResponse(bboxes=[BboxOut(**b) for b in bboxes])


@app.post("/inpaint_and_render", response_model=InpaintAndRenderResponse)
async def inpaint_and_render(
    file: UploadFile = File(...),
    translations: str = Form(""),
    bboxes: str = Form(""),
    data_id: str = Form(""),
    font_dir: str = Form(""),
    debug: str = Form(""),
    link_threshold: str = Form(""),
):
    """擦除指定 bbox 并嵌字"""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "仅支持图片文件")

    if not translations.strip():
        raise HTTPException(400, "translations 参数不能为空")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"无法读取图片: {e}")

    img_array = np.array(image)

    # 1. 解析翻译数据
    import json
    try:
        trans_list = json.loads(translations)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"translations JSON 格式错误: {e}")

    translations_obj = [RenderTranslation(**t) for t in trans_list]

    # 2. 收集需要擦除的 bbox IDs
    all_erase_ids: set = set()
    for t in trans_list:
        ids = t.get("bbox_ids") or [t.get("bbox_id", 0)]
        for bid in ids:
            all_erase_ids.add(bid)

    # 3. 获取 bbox：优先 data_id，其次 bboxes，最后自动检测
    if data_id.strip():
        session = _get_session(data_id)
        all_bboxes = session["bboxes"]
        logger.info(f"使用缓存 {data_id}: {len(all_bboxes)} 区域")
    elif bboxes.strip():
        parsed = json.loads(bboxes)
        if isinstance(parsed, dict):
            if "bboxes" in parsed:
                all_bboxes = parsed["bboxes"]
            elif "results" in parsed:
                all_bboxes = parsed["results"]
            else:
                raise HTTPException(400, "bboxes 格式错误")
        elif isinstance(parsed, list):
            all_bboxes = parsed
        else:
            raise HTTPException(400, "bboxes 格式错误")
    else:
        lt = float(link_threshold) if link_threshold.strip() else 0
        all_bboxes = detect_text(img_array, link_threshold=lt)
    # 统一 bbox 键名：OCR 返回 bbox_id，CRAFT 返回 id
    for b in all_bboxes:
        if "bbox_id" in b and "id" not in b:
            b["id"] = b["bbox_id"]
    bbox_map = {b["id"]: b for b in all_bboxes}

    # 4. 只擦除翻译中引用的 bbox
    erase_bboxes = [bbox_map[i] for i in all_erase_ids if i in bbox_map]
    logger.info(f"擦除 {len(erase_bboxes)}/{len(all_bboxes)} 区域 (IDs: {sorted(all_erase_ids)})")

    if erase_bboxes:
        img_array = inpaint(img_array, erase_bboxes)

    # 5. 嵌字渲染
    fd = Path(font_dir) if font_dir else FONT_DIR
    rid = uuid.uuid4().hex[:8]
    inpainted_path = f"/tmp/_inpainted_{rid}.png"
    output_path = f"/tmp/_rendered_{rid}.png"
    Image.fromarray(img_array).save(inpainted_path)

    # 只传 Bbox 需要的字段
    render_bboxes = [RenderBbox(
        id=b.get("id", b.get("bbox_id", 0)),
        x=b["x"], y=b["y"], w=b["w"], h=b["h"],
        polygon=b.get("polygon", []),
    ) for b in all_bboxes]
    do_render(inpainted_path, render_bboxes, translations_obj, output_path, fd)

    # Debug：在渲染结果上绘制 bbox 边框
    if debug.strip() or app.state.debug:
        from PIL import ImageDraw as PILDraw
        debug_img = Image.open(output_path).convert("RGBA")
        debug_draw = PILDraw.Draw(debug_img)
        colors = ["#ff4444","#44cc44","#4488ff","#ffaa00","#ff44ff",
                  "#44ffff","#88ff88","#ff8888","#aacc44","#cc44cc"]
        color_idx = 0
        for t in trans_list:
            ids = t.get("bbox_ids") or [t.get("bbox_id", 0)]
            color = colors[color_idx % len(colors)]
            color_idx += 1
            for bid in ids:
                if bid in bbox_map:
                    b = bbox_map[bid]
                    debug_draw.rectangle(
                        [b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]],
                        outline=color, width=2)
                    debug_draw.text((b["x"]+2, b["y"]+2), str(bid), fill=color)
        debug_img.convert("RGB").save(output_path)

    # 6. 返回
    with open(output_path, "rb") as f:
        result_b64 = base64.b64encode(f.read()).decode()

    return InpaintAndRenderResponse(rendered_image=result_b64)


@app.post("/ocr")
async def ocr_bboxes(
    file: UploadFile = File(...),
    bboxes: str = Form(""),
    link_threshold: str = Form(""),
):
    """对指定 bbox 区域做 OCR，返回精确文字"""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "仅支持图片文件")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"无法读取图片: {e}")

    # 解析 bbox 或自动检测
    if bboxes.strip():
        parsed = json.loads(bboxes)
        if isinstance(parsed, dict) and "bboxes" in parsed:
            bbox_list = parsed["bboxes"]
        elif isinstance(parsed, list):
            bbox_list = parsed
        else:
            raise HTTPException(400, "bboxes 格式错误")
    else:
        lt = float(link_threshold) if link_threshold.strip() else 0
        bbox_list = detect_text(np.array(image), link_threshold=lt)

    load_ocr()
    results = []
    for b in bbox_list:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        # 留少许边距避免裁到文字边缘
        pad = 4
        crop = image.crop((
            max(0, x - pad),
            max(0, y - pad),
            min(image.width, x + w + pad),
            min(image.height, y + h + pad),
        ))
        try:
            text = manga_ocr(crop)
        except Exception as e:
            text = f"[OCR ERROR: {e}]"
        results.append({
            "bbox_id": b["id"],
            "text": text.strip(),
            "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
            "polygon": b.get("polygon", []),
        })

    data_id = _save_session(bboxes=bbox_list, results=results)
    return {"status": "ok", "results": results, "data_id": data_id}


@app.get("/health")
async def health():
    return {"status": "ok", "craft_loaded": craft_net is not None, "lama_loaded": lama_model is not None}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true", help="debug 模式：渲染时绘制 bbox 边框")
    p.add_argument("--link-threshold", type=float, default=DEFAULT_LINK_THRESHOLD, help=f"CRAFT link_threshold (默认: {DEFAULT_LINK_THRESHOLD})")
    args, _ = p.parse_known_args()
    if args.debug:
        logger.info("DEBUG 模式已启用")
        app.state.debug = True
    else:
        app.state.debug = False
    app.state.link_threshold = args.link_threshold
    logger.info(f"默认 link_threshold={app.state.link_threshold}")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8899)
