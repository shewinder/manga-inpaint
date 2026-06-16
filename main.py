"""
manga-inpaint: 漫画文字检测 + 擦除 + 嵌字服务

POST /detect            CRAFT 检测 → bbox 列表
POST /inpaint_and_render  擦除 + 嵌字 → 结果图片
"""
import io
import os
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

# ---- 配置 ----
CONFIDENCE_THRESHOLD = 0.7
MASK_DILATION = 6
INPAINT_MAX_SIZE = 2048
BASE_DIR = Path(__file__).parent
FONT_DIR = BASE_DIR / "fonts"
MODEL_DIR = BASE_DIR / "models"

# 项目本地模型路径
CRAFT_WEIGHT_PATH = str(MODEL_DIR / "craft" / "craft_mlt_25k.pth")
REFINER_WEIGHT_PATH = str(MODEL_DIR / "craft" / "craft_refiner_CTW1500.pth")
LAMA_MODEL_PATH = str(MODEL_DIR / "lama" / "big-lama.pt")


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
    font: str = "sans"
    color: str = "#000000"
    direction: str = "auto"
    outline: bool = True
    outline_color: str = "#ffffff"
    bbox_ids: Optional[List[int]] = None


class InpaintAndRenderRequest(BaseModel):
    translations: List[TranslationIn]
    font_dir: str = ""


class InpaintAndRenderResponse(BaseModel):
    rendered_image: str  # base64


# ---- 核心函数 ----

def load_craft():
    global craft_net, refine_net
    if craft_net is None:
        import torchvision.models.vgg as vgg
        if not hasattr(vgg, 'model_urls'):
            vgg.model_urls = {
                "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            }
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
        logger.info("加载 LaMa 模型...")
        os.environ["LAMA_MODEL"] = LAMA_MODEL_PATH
        from simple_lama_inpainting import SimpleLama
        lama_model = SimpleLama()
        logger.info("LaMa 模型加载完成")


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


def detect_text(image: np.ndarray) -> List[dict]:
    from craft_text_detector import get_prediction, empty_cuda_cache

    load_craft()

    logger.info(f"开始文字检测, 图片尺寸: {image.shape}")

    np.array = _compat_array
    try:
        prediction = get_prediction(
            image=image,
            craft_net=craft_net,
            refine_net=refine_net,
            text_threshold=CONFIDENCE_THRESHOLD,
            link_threshold=0.4,
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
async def detect(file: UploadFile = File(...)):
    """CRAFT 文字检测 — 只检测，不擦除"""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "仅支持图片文件")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"无法读取图片: {e}")

    img_array = np.array(image)
    bboxes = detect_text(img_array)
    return DetectResponse(bboxes=[BboxOut(**b) for b in bboxes])


@app.post("/inpaint_and_render", response_model=InpaintAndRenderResponse)
async def inpaint_and_render(
    file: UploadFile = File(...),
    translations: str = Form(""),
    bboxes: str = Form(""),
    font_dir: str = Form(""),
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

    # 3. 解析传入的 bbox（允许带坐标，避免重复检测）
    if bboxes.strip():
        parsed = json.loads(bboxes)
        # 兼容两种格式：纯数组 或 detect 返回的 {"success":true, "bboxes":[...]}
        if isinstance(parsed, dict) and "bboxes" in parsed:
            all_bboxes = parsed["bboxes"]
        elif isinstance(parsed, list):
            all_bboxes = parsed
        else:
            raise HTTPException(400, "bboxes 格式错误，需要数组或 {bboxes:[...]}")
    else:
        all_bboxes = detect_text(img_array)
    bbox_map = {b["id"]: b for b in all_bboxes}

    # 4. 只擦除翻译中引用的 bbox
    erase_bboxes = [bbox_map[i] for i in all_erase_ids if i in bbox_map]
    logger.info(f"擦除 {len(erase_bboxes)}/{len(all_bboxes)} 区域 (IDs: {sorted(all_erase_ids)})")

    if erase_bboxes:
        img_array = inpaint(img_array, erase_bboxes)

    # 5. 嵌字渲染
    fd = Path(font_dir) if font_dir else FONT_DIR
    inpainted_path = "/tmp/_inpainted_temp.png"
    output_path = "/tmp/_rendered_temp.png"
    Image.fromarray(img_array).save(inpainted_path)

    render_bboxes = [RenderBbox(**b) for b in all_bboxes]
    do_render(inpainted_path, render_bboxes, translations_obj, output_path, fd)

    # 6. 返回
    with open(output_path, "rb") as f:
        result_b64 = base64.b64encode(f.read()).decode()

    return InpaintAndRenderResponse(rendered_image=result_b64)


@app.get("/health")
async def health():
    return {"status": "ok", "craft_loaded": craft_net is not None, "lama_loaded": lama_model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8899)
