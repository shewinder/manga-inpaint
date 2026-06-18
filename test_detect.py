"""直接调用 detect_text 做 link_threshold 扫描，不经过 HTTP"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from main import detect_text, load_craft, _compat_array, _original_array

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
COLORS = ["#ff4444", "#44ff44", "#4488ff", "#ffaa00", "#ff44ff",
          "#44ffff", "#ff8888", "#88ff88", "#aacc44", "#cc44cc",
          "#ff8844", "#8844ff", "#44ff88", "#ff4488", "#88ff44"]


def draw_bboxes(image: Image.Image, bboxes: list) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for i, b in enumerate(bboxes):
        color = COLORS[i % len(COLORS)]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        # polygon
        poly = b.get("polygon", [])
        if poly and len(poly) >= 4:
            draw.polygon([(p[0], p[1]) for p in poly], outline=color, width=1)
        # label: id + size
        label = f"#{i} {w}x{h}"
        tw = draw.textlength(label) if hasattr(draw, 'textlength') else len(label) * 7
        draw.rectangle([x, y - 20, x + tw + 4, y], fill=color)
        draw.text((x + 2, y - 18), label, fill="white")
    return image


def main():
    test_dir = Path("test")
    output_dir = test_dir / "output"
    output_dir.mkdir(exist_ok=True)

    images = sorted(
        p for p in test_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
        and "_" not in p.stem  # skip annotated outputs
    )

    if not images:
        print(f"{test_dir} 中没有图片")
        sys.exit(1)

    # 加载模型
    print("加载 CRAFT 模型...")
    load_craft()

    thresholds = [round(x, 2) for x in np.arange(0.10, 0.41, 0.05)]
    summary = {}

    for img_path in images:
        name = img_path.stem
        summary[name] = {}
        print(f"\n{'='*60}")
        print(f"图片: {img_path.name}")
        print(f"{'='*60}")

        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        for lt in thresholds:
            tag = f"link{lt:.2f}"
            print(f"  lt={lt:.2f} ...", end=" ", flush=True)

            bboxes = detect_text(img_array, link_threshold=lt)

            # 统计
            areas = [b["w"] * b["h"] for b in bboxes]
            avg_area = sum(areas) // max(len(areas), 1)
            max_box = max(bboxes, key=lambda b: b["w"] * b["h"])
            min_box = min(bboxes, key=lambda b: b["w"] * b["h"])
            summary[name][tag] = {
                "count": len(bboxes),
                "max": f"{max_box['w']}x{max_box['h']}",
                "min": f"{min_box['w']}x{min_box['h']}",
                "avg_area": avg_area,
                "ratios": [round(b["w"] / max(b["h"], 1), 3) for b in bboxes],
            }

            print(f"{len(bboxes)} 框  max={summary[name][tag]['max']}  avg={avg_area}")

            # 画 bbox 图
            out_img = img.copy()
            draw_bboxes(out_img, bboxes)
            out_img.save(output_dir / f"{name}_{tag}_bboxes.png")

    # 保存汇总
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总: {summary_path}")

    # 对比表
    print(f"\n{'图片':<10}", end="")
    for lt in thresholds:
        print(f" lt={lt:.2f} ", end="")
    print()
    for name, data in summary.items():
        print(f"{name:<10}", end="")
        for lt in thresholds:
            tag = f"link{lt:.2f}"
            print(f" {data[tag]['count']:>5} ", end="")
        print()
    print()


if __name__ == "__main__":
    main()
