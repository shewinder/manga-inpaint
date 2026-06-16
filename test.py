"""manga-inpaint 批量测试脚本

用法:
  python test.py test                    # 批量检测+擦除所有图片
  python test.py test --detect-only       # 只检测，画 bbox
  python test.py test --draw-bboxes       # 检测+擦除 + 画 bbox
  python test.py test --render            # 检测+擦除+嵌字（用模拟翻译数据）
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw

SERVICE_URL = "http://localhost:8899"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def draw_bboxes_on_image(image: Image.Image, bboxes: list, output_path: Path):
    draw = ImageDraw.Draw(image)
    colors = ["#ff4444", "#44ff44", "#4488ff", "#ffaa00", "#ff44ff",
              "#44ffff", "#ff8888", "#88ff88"]
    for i, b in enumerate(bboxes):
        color = colors[i % len(colors)]
        draw.rectangle(
            [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]],
            outline=color, width=3,
        )
        poly = b.get("polygon", [])
        if poly and len(poly) >= 4:
            draw.polygon([(p[0], p[1]) for p in poly], outline=color, width=1)
        label = f"{b['id']}"
        draw.rectangle([b["x"], b["y"] - 22, b["x"] + 60, b["y"]], fill=color)
        draw.text((b["x"] + 2, b["y"] - 20), label, fill="white")
    image.save(output_path, "PNG")


def detect(path: Path) -> list | None:
    """调用 /detect"""
    with open(path, "rb") as f:
        r = requests.post(
            f"{SERVICE_URL}/detect",
            files={"file": (path.name, f, f"image/{path.suffix.lstrip('.')}")},
            timeout=300,
        )
    if r.status_code != 200:
        print(f"  ✗ /detect HTTP {r.status_code}")
        return None
    return r.json()["bboxes"]


def inpaint_and_render(path: Path, translations: list, output_path: Path) -> str | None:
    """调用 /inpaint_and_render"""
    with open(path, "rb") as f:
        r = requests.post(
            f"{SERVICE_URL}/inpaint_and_render",
            files={"file": (path.name, f, f"image/{path.suffix.lstrip('.')}")},
            data={"translations": json.dumps(translations, ensure_ascii=False)},
            timeout=300,
        )
    if r.status_code != 200:
        print(f"  ✗ /inpaint_and_render HTTP {r.status_code}: {r.text[:200]}")
        return None
    data = r.json()
    img_bytes = base64.b64decode(data["rendered_image"])
    img = Image.open(io.BytesIO(img_bytes))
    img.save(output_path, "PNG")
    return f"{img.size}"


def make_mock_translations(bboxes: list) -> list:
    """生成模拟翻译数据"""
    texts = ["你好世界", "今天天气不错", "对话框文字", "测试一二三",
             "竖排文字", "漫画翻译", "嵌字功能", "试试看",
             "可以的", "横排文字", "竖排测试", "日漫",
             "翻译测试中", "这里是对白", "气泡内部文字",
             "拟声词砰砰", "最后一", "个测试"]
    result = []
    for i, b in enumerate(bboxes):
        direction = "vertical" if b["h"] > b["w"] * 1.5 else "horizontal"
        result.append({
            "bbox_id": i,
            "text": texts[i % len(texts)],
            "font": "sans",
            "color": "#000000",
            "direction": direction,
            "outline": True,
            "outline_color": "#ffffff",
        })
    return result


def main():
    parser = argparse.ArgumentParser(description="manga-inpaint 批量测试")
    parser.add_argument("dir", nargs="?", default="test", help="测试目录 (默认: test)")
    parser.add_argument("--detect-only", action="store_true", help="只检测，不擦除")
    parser.add_argument("--draw-bboxes", action="store_true", help="在图上绘制 bbox")
    parser.add_argument("--render", action="store_true", help="擦除+嵌字（模拟翻译）")
    args = parser.parse_args()

    test_dir = Path(args.dir)
    if not test_dir.is_dir():
        print(f"目录不存在: {test_dir}")
        sys.exit(1)

    images = sorted(
        p for p in test_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
        and "_inpainted" not in p.stem
        and "_bboxes" not in p.stem
        and "_rendered" not in p.stem
    )
    if not images:
        print(f"{test_dir} 中没有图片")
        sys.exit(0)

    output_dir = test_dir / "output"
    output_dir.mkdir(exist_ok=True)

    mode = []
    if args.detect_only:
        mode.append("只检测")
    if args.render:
        mode.append("嵌字")
    if args.draw_bboxes:
        mode.append("画框")

    print(f"测试目录: {test_dir} ({len(images)} 张)")
    print(f"模式: {'+'.join(mode) if mode else '检测+擦除'}")
    print(f"服务: {SERVICE_URL}")
    print("-" * 60)

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}", end=" ")
        sys.stdout.flush()

        # Step 1: 检测
        bboxes = detect(img_path)
        if bboxes is None:
            continue
        print(f"- {len(bboxes)} bbox", end="")

        if args.draw_bboxes:
            src_img = Image.open(img_path).convert("RGB")
            draw_bboxes_on_image(src_img, bboxes,
                                 output_dir / f"{img_path.stem}_bboxes.png")
            print("+box", end="")

        if args.detect_only:
            print()
            continue

        # Step 2: 擦除 + 嵌字
        if args.render:
            translations = make_mock_translations(bboxes)
        else:
            # 不翻译，只擦除所有 bbox
            translations = []
            for b in bboxes:
                translations.append({
                    "bbox_id": b["id"],
                    "text": "●",  # 占位符
                    "font": "sans",
                    "color": "#000000",
                    "direction": "vertical" if b["h"] > b["w"] * 1.5 else "horizontal",
                    "outline": False,
                })

        ext = "_rendered.png" if args.render else "_inpainted.png"
        result = inpaint_and_render(img_path, translations,
                                     output_dir / f"{img_path.stem}{ext}")
        if result:
            size_mb = (output_dir / f"{img_path.stem}{ext}").stat().st_size / 1024 / 1024
            print(f" → {result} {size_mb:.1f}MB")

    print("-" * 60)
    print("完成")


if __name__ == "__main__":
    main()
