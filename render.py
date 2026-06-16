"""
漫画文字渲染脚本
用法: python render.py --inpainted <擦除后图片> --translations '<JSON>' --bboxes '<JSON>' --output <结果>
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# 字体映射 — AI 传入字体代号，脚本匹配实际文件
FONT_MAP = {
    "sans": "simhei.ttf",
    "heiti": "msyh.ttc",   # 微软雅黑，符号(♡♪★等)渲染更好
    "serif": "simsun.ttc",
    "bold": "msyh.ttc",
}
DEFAULT_FONT = "sans"


@dataclass
class Bbox:
    id: int
    x: int
    y: int
    w: int
    h: int
    polygon: List[List[float]]


@dataclass
class Translation:
    text: str
    font: str = "sans"
    color: str = "#000000"
    direction: str = "auto"  # horizontal | vertical | auto
    outline: bool = True
    outline_color: str = "#ffffff"
    bbox_id: int = 0  # 主 bbox，单框时必填
    bbox_ids: Optional[List[int]] = None  # 覆盖多个 bbox 时指定全部 id


def load_translations(data: list) -> List[Translation]:
    return [Translation(**t) for t in data]


def load_bboxes(data: list) -> List[Bbox]:
    return [Bbox(**b) for b in data]


def find_font(font_dir: Path, name: str) -> Optional[Path]:
    """查找 ttf/otf 字体文件，支持模糊匹配"""
    if not font_dir.is_dir():
        return None
    for f in sorted(font_dir.iterdir()):
        fname = f.name.lower()
        if name.lower() in fname and f.suffix.lower() in (".ttf", ".otf", ".ttc"):
            return f
    return None


def load_font(font_dir: Path, size: int, font_name: str = "sans") -> ImageFont.FreeTypeFont:
    """加载字体，找不到则 fallback 到 default"""
    key = FONT_MAP.get(font_name, FONT_MAP[DEFAULT_FONT])
    path = find_font(font_dir, key)
    if path:
        return ImageFont.truetype(str(path), size)
    # 尝试系统字体
    fallbacks = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf",
    ]
    for fb in fallbacks:
        if Path(fb).exists():
            return ImageFont.truetype(fb, size)
    # 最后兜底：Pillow 默认字体
    return ImageFont.load_default()


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0, 0, 0)
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def determine_direction(polygon: List[List[float]]) -> str:
    """从 bbox 宽高比和 polygon 变化判断文字方向"""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    # 竖排版（日文漫画常见）：bbox 高明显大于宽
    if h > w * 1.5:
        return "vertical"
    return "horizontal"


def calculate_angle(polygon: List[List[float]]) -> float:
    """计算文字区域的旋转角度（度）"""
    if len(polygon) < 4:
        return 0.0
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    # 取最小 x 的两个点，计算左侧边的角度
    points_sorted = sorted(zip(xs, ys), key=lambda t: t[0])
    left_bottom = points_sorted[0]
    left_top = min(points_sorted[:2], key=lambda t: t[1])
    dx = left_bottom[0] - left_top[0]
    dy = left_bottom[1] - left_top[1]
    if abs(dx) < 1:
        return 0.0
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def fit_font_size(
    draw: ImageDraw.Draw,
    text: str,
    max_w: int,
    max_h: int,
    font_dir: Path,
    font_name: str,
    direction: str,
    padding: int = 8,
) -> int:
    """二分查找最大可用字号，测量与实际渲染逻辑一致"""
    avail_w = max_w - padding * 2
    avail_h = max_h - padding * 2
    lo, hi = 8, max(avail_w, avail_h)
    best = lo

    for _ in range(20):
        mid = (lo + hi) // 2
        font = load_font(font_dir, mid, font_name)
        ascent, descent = font.getmetrics()
        line_h = int((ascent + descent) * 1.15)

        if direction == "vertical":
            cws = [draw.textbbox((0, 0), c, font=font, anchor="lt")[2] for c in text]
            tw = max(cws) if cws else 0
            one_col_h = line_h * len(text)
            if one_col_h <= avail_h:
                # 单列能放下
                th = one_col_h + (draw.textbbox((0, 0), text[0] if text else " ", font=font, anchor="lt")[3] - line_h)
            else:
                # 需要多列：列数 = ceil(total_h / avail_h)
                chars_per_col = max(1, avail_h // line_h)
                num_cols = (len(text) + chars_per_col - 1) // chars_per_col
                th = avail_h
                tw = num_cols * tw + (num_cols - 1) * 8  # 8px 列间距
        else:
            lines = _wrap_text(draw, text, font, avail_w)
            lws = []
            total_th = 0
            for i, l in enumerate(lines):
                b = draw.textbbox((0, 0), l, font=font, anchor="lt")
                lws.append(b[2])
                if i == len(lines) - 1:
                    total_th += b[3]
                else:
                    total_th += line_h
            th = total_th
            tw = max(lws) if lws else 0

        if tw <= avail_w and th <= avail_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def draw_vertical_text(
    draw: ImageDraw.Draw,
    top_left: Tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int],
    outline_width: int,
    outline_fill: Tuple[int, int, int],
    max_h: int = 0,
    max_w: int = 0,
    line_spacing: float = 1.15,
    col_gap: int = 8,
):
    """逐字绘制竖排文字，lt 锚点，从上到下。高度溢出时自动拆多列"""
    x0, y0 = top_left
    ascent, descent = font.getmetrics()
    char_step = int((ascent + descent) * line_spacing)

    # 每字尺寸
    char_sizes = []
    for c in text:
        b = draw.textbbox((0, 0), c, font=font, anchor="lt")
        char_sizes.append((b[2], b[3]))

    if not char_sizes:
        return
    char_w = max(s[0] for s in char_sizes)

    # 计算需要多少列
    one_col_h = char_step * len(text)
    if max_h and one_col_h > max_h:
        col_h = max_h
        chars_per_col = max(1, max_h // char_step)
        num_cols = (len(text) + chars_per_col - 1) // chars_per_col
    else:
        col_h = one_col_h
        chars_per_col = len(text)
        num_cols = 1

    total_w = num_cols * char_w + (num_cols - 1) * col_gap

    # 水平居中（多列时从右到左排列）
    if max_w and total_w < max_w:
        x0 += (max_w - total_w) // 2

    for col in range(num_cols):
        start_idx = col * chars_per_col
        col_text = text[start_idx:start_idx + chars_per_col]

        # 日文竖排从右到左：最后一列在最右边
        col_x = x0 + (num_cols - 1 - col) * (char_w + col_gap)

        # 垂直居中
        col_total_h = char_step * (len(col_text) - 1) + (char_sizes[min(start_idx, len(char_sizes)-1)][1])
        cy = y0 + (col_h - col_total_h) // 2 if max_h else y0

        for j, char in enumerate(col_text):
            idx = start_idx + j
            cw = char_sizes[idx][0]
            cx = col_x + (char_w - cw) // 2

            if outline_width > 0:
                _draw_outlined_text(draw, char, font, cx, cy, fill,
                                    outline_width, outline_fill, anchor="lt")
            else:
                draw.text((cx, cy), char, font=font, fill=fill, anchor="lt")
            cy += char_step


def draw_horizontal_text(
    draw: ImageDraw.Draw,
    top_left: Tuple[int, int],
    max_size: Tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int],
    outline_width: int,
    outline_fill: Tuple[int, int, int],
    rotation: float = 0.0,
    padding: int = 4,
):
    """绘制横排文字（居中），支持旋转"""
    x0, y0 = top_left
    mw, mh = max_size
    mw -= padding * 2
    mh -= padding * 2
    x0 += padding
    y0 += padding
    ascent, descent = font.getmetrics()
    line_step = int((ascent + descent) * 1.15)

    lines = _wrap_text(draw, text, font, mw)

    # 测量：用 lt 锚点，bbox 即 (0,0) 到 (w,h)
    line_metrics = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, anchor="lt")
        line_metrics.append((bbox[2], bbox[3]))  # w, h
    total_w = max(m[0] for m in line_metrics) if line_metrics else 0
    total_h = line_step * (len(lines) - 1) + (line_metrics[-1][1] if line_metrics else line_step)

    # 垂直居中
    start_y = y0 + (mh - total_h) // 2

    if abs(rotation) < 0.5:
        rotation = 0.0

    if abs(rotation) > 0.5:
        _draw_horizontal_rotated(draw, lines, font, fill, outline_width,
                                  outline_fill, x0, y0, mw + padding * 2,
                                  mh + padding * 2, rotation,
                                  start_y, total_w, total_h)
    else:
        cy = start_y
        for i, line in enumerate(lines):
            lw, lh = line_metrics[i]
            cx = x0 + (mw - lw) // 2
            if outline_width > 0:
                _draw_outlined_text(draw, line, font, cx, cy, fill,
                                     outline_width, outline_fill, anchor="lt")
            else:
                draw.text((cx, cy), line, font=font, fill=fill, anchor="lt")
            cy += line_step


def _wrap_text(draw, text: str, font, max_w: int) -> List[str]:
    """简单换行：按字符填充"""
    lines = []
    current = ""
    for ch in text:
        test = current + ch
        bbox = draw.textbbox((0, 0), test, font=font)
        w = bbox[2] - bbox[0]
        if w > max_w and current:
            lines.append(current)
            current = ch
        else:
            current = test
    if current:
        lines.append(current)
    return lines or [text]


def _draw_outlined_text(draw, text, font, x, y, fill, ow, ofill, anchor="lt"):
    """描边文字"""
    for dx in range(-ow, ow + 1):
        for dy in range(-ow, ow + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=ofill, anchor=anchor)
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)


def _draw_horizontal_rotated(draw, lines, font, fill, ow, ofill,
                               x0, y0, mw, mh, angle, start_y, tw, th):
    """在旋转画布上绘制横排文字"""
    import numpy as np
    from PIL import Image as PILImage

    # 创建临时层
    txt_img = PILImage.new("RGBA", (mw, mh), (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_img)

    cy = start_y
    for line in lines:
        line_bbox = txt_draw.textbbox((0, 0), line, font=font, anchor="lt")
        lw = line_bbox[2]
        cx = (mw - lw) // 2
        if ow > 0:
            _draw_outlined_text(txt_draw, line, font, cx, cy,
                                fill + (255,), ow, ofill + (255,), anchor="lt")
        else:
            txt_draw.text((cx, cy), line, font=font, fill=fill + (255,), anchor="lt")
        cy += txt_draw.textbbox((0, 0), line, font=font, anchor="lt")[3]

    # 旋转并粘贴
    rotated = txt_img.rotate(-angle, expand=False, resample=PILImage.BICUBIC)
    draw._image.paste(rotated, (x0, y0), rotated)


def merge_overlapping_bboxes(
    bboxes: List[Bbox],
    translations: List[Translation],
    overlap_ratio: float = 0.2,
) -> Tuple[List[Bbox], List[Translation]]:
    """合并重叠面积大于阈值的相邻 bbox"""
    n = len(bboxes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    # 检测重叠并合并
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = bboxes[i], bboxes[j]
            ox = max(0, min(bi.x + bi.w, bj.x + bj.w) - max(bi.x, bj.x))
            oy = max(0, min(bi.y + bi.h, bj.y + bj.h) - max(bi.y, bj.y))
            if ox > 0 and oy > 0:
                area_i, area_j = bi.w * bi.h, bj.w * bj.h
                overlap_area = ox * oy
                if overlap_area > min(area_i, area_j) * overlap_ratio:
                    union(i, j)

    # 按组聚合
    groups: dict[int, list] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    if len(groups) == n:
        return bboxes, translations  # 无合并

    # 生成合并后的 bbox 和 translation
    trans_map = {t.bbox_id: t for t in translations}
    new_bboxes = []
    new_translations = []

    for indices in groups.values():
        if len(indices) == 1:
            i = indices[0]
            new_bboxes.append(bboxes[i])
            if i in trans_map:
                new_translations.append(trans_map[i])
        else:
            # 合并 bbox：取包围盒
            merged = Bbox(
                id=bboxes[indices[0]].id,
                x=min(bboxes[i].x for i in indices),
                y=min(bboxes[i].y for i in indices),
                w=max(bboxes[i].x + bboxes[i].w for i in indices) - min(bboxes[i].x for i in indices),
                h=max(bboxes[i].y + bboxes[i].h for i in indices) - min(bboxes[i].y for i in indices),
                polygon=[],
            )
            new_bboxes.append(merged)

            # 合并翻译文本
            texts = []
            for i in sorted(indices):
                if i in trans_map:
                    texts.append(trans_map[i].text)
            merged_text = "".join(texts)
            base_trans = trans_map.get(indices[0], translations[0])
            new_translations.append(Translation(
                bbox_id=merged.id,
                text=merged_text,
                font=base_trans.font,
                color=base_trans.color,
                direction=base_trans.direction,
                outline=base_trans.outline,
                outline_color=base_trans.outline_color,
            ))

    print(f"  bbox 合并: {len(bboxes)} → {len(new_bboxes)} (合并了 {len(bboxes)-len(new_bboxes)} 组)")
    return new_bboxes, new_translations


def render(
    inpainted_path: str,
    bboxes: List[Bbox],
    translations: List[Translation],
    output_path: str,
    font_dir: Path,
) -> None:
    img = Image.open(inpainted_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # 建立 bbox_id → bbox 索引
    bbox_map = {b.id: b for b in bboxes}
    rendered_bboxes: set = set()  # 已渲染的 bbox，避免重复

    for trans in translations:
        # 确定覆盖的 bbox 列表，排除已被前序 translation 渲染的
        bbox_ids = trans.bbox_ids if trans.bbox_ids else [trans.bbox_id]
        bbox_ids = [i for i in bbox_ids if i not in rendered_bboxes]
        # 确定覆盖的 bbox 列表
        bbox_ids = trans.bbox_ids if trans.bbox_ids else [trans.bbox_id]
        matched = [bbox_map[i] for i in bbox_ids if i in bbox_map]

        if not matched:
            print(f"  ⚠ bbox_ids={bbox_ids} 均不存在，跳过")
            continue

        # 取所有匹配 bbox 的并集作为渲染区域
        x0 = min(b.x for b in matched)
        y0 = min(b.y for b in matched)
        x1 = max(b.x + b.w for b in matched)
        y1 = max(b.y + b.h for b in matched)
        union_w = x1 - x0
        union_h = y1 - y0
        union_poly = matched[0].polygon  # 方向判断用第一个

        # 确定文字方向
        direction = trans.direction
        if direction == "auto":
            direction = determine_direction(union_poly)

        # 计算旋转角
        angle = calculate_angle(union_poly)

        # 找最佳字号
        font_size = fit_font_size(draw, trans.text, union_w, union_h,
                                   font_dir, trans.font, direction)
        font = load_font(font_dir, font_size, trans.font)

        # 颜色
        fill_rgb = hex_to_rgb(trans.color)
        outline_width = max(1, font_size // 10) if trans.outline else 0
        outline_rgb = hex_to_rgb(trans.outline_color)

        ids_str = f"bboxes={bbox_ids}" if len(bbox_ids) > 1 else f"bbox_id={trans.bbox_id}"
        print(f"  [{ids_str}] \"{trans.text[:15]}...\" "
              f"{direction} {union_w}x{union_h} size={font_size} color={trans.color}")

        rendered_bboxes.update(bbox_ids)

        if direction == "vertical":
            draw_vertical_text(draw, (x0, y0), trans.text,
                                font, fill_rgb, outline_width, outline_rgb,
                                max_h=union_h, max_w=union_w)
        else:
            draw_horizontal_text(draw, (x0, y0),
                                  (union_w, union_h), trans.text,
                                  font, fill_rgb, outline_width,
                                  outline_rgb, angle)

    # 保存
    img.convert("RGB").save(output_path, "PNG")
    print(f"  → 保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="漫画文字嵌字渲染")
    parser.add_argument("--inpainted", required=True, help="擦除后的图片路径")
    parser.add_argument("--bboxes", required=True,
                        help="bbox 列表 JSON (来自 GPU 服务)")
    parser.add_argument("--translations", required=True,
                        help="翻译数据 JSON (包含渲染参数)")
    parser.add_argument("--output", required=True, help="输出图片路径")
    parser.add_argument("--font-dir", default="fonts",
                        help="字体目录 (默认: fonts/)")
    args = parser.parse_args()

    bboxes = load_bboxes(json.loads(args.bboxes))
    translations = load_translations(json.loads(args.translations))
    font_dir = Path(args.font_dir)

    if not font_dir.is_dir():
        print(f"字体目录不存在: {font_dir}")
        print("请创建 fonts/ 目录并放入中文字体文件（如 思源黑体）")
        sys.exit(1)

    render(args.inpainted, bboxes, translations, args.output, font_dir)


if __name__ == "__main__":
    main()
