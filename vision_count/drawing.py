"""Vẽ khung và nhãn lên ảnh. Tách riêng để giao diện nào cũng dùng được."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from vision_count.detector import Detection

# Bảng màu: mỗi loại vật (class_id) lấy một màu cố định để dễ phân biệt
PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
    (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
    (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

# Màu vàng cho vùng đếm, khác hẳn màu các khung vật thể
REGION_COLOR = (255, 214, 0)

# Font có đủ dấu tiếng Việt. Font mặc định của Pillow KHÔNG có, sẽ vẽ dấu thành ô vuông.
# Pillow tự tìm các tên này trong thư mục font của hệ điều hành.
VIETNAMESE_FONTS = [
    "Arial.ttf",  # macOS, Windows
    "arial.ttf",  # Windows (tên chữ thường)
    "DejaVuSans.ttf",  # Linux
    "LiberationSans-Regular.ttf",  # Linux
    "NotoSans-Regular.ttf",  # Linux
]


@lru_cache(maxsize=1)
def _find_font_name() -> str | None:
    """Tìm (một lần) tên font hỗ trợ tiếng Việt có trên máy; không có thì trả về None."""
    for name in VIETNAMESE_FONTS:
        try:
            ImageFont.truetype(name, 10)
            return name
        except OSError:
            continue
    return None


@lru_cache(maxsize=16)
def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Font hỗ trợ tiếng Việt ở cỡ chữ size; không có thì dùng font mặc định (mất dấu nhưng không lỗi)."""
    name = _find_font_name()
    return ImageFont.truetype(name, size) if name else ImageFont.load_default(size=size)


# Kiểu nhãn trên ảnh. Khi vật dày đặc, nhãn đầy đủ sẽ đè lên nhau, nên có thêm 2 kiểu gọn.
LABEL_FULL = "full"  # "#1 người 87%"
LABEL_NUMBER = "number"  # "1"
LABEL_NONE = "none"  # chỉ vẽ khung
LABEL_STYLES = (LABEL_FULL, LABEL_NUMBER, LABEL_NONE)


def _line_width(image: Image.Image) -> int:
    """Độ dày nét tỉ lệ theo kích thước ảnh, để ảnh to hay nhỏ đều dễ nhìn."""
    return max(2, round(3 * max(image.size) / 1000))


def draw_region(
    image: Image.Image,
    region: tuple[float, float, float, float] | None = None,
    point: tuple[float, float] | None = None,
) -> Image.Image:
    """Vẽ vùng đếm (hình chữ nhật vàng) và/hoặc một điểm đánh dấu lên BẢN SAO của ảnh."""
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    width = _line_width(marked)
    if region is not None:
        draw.rectangle(region, outline=REGION_COLOR, width=width)
    if point is not None:
        r = width * 3
        x, y = point
        draw.ellipse((x - r, y - r, x + r, y + r), fill=REGION_COLOR)
    return marked


def draw_detections(
    image: Image.Image,
    detections: list[Detection],
    label_fn: Callable[[str], str] | None = None,
    label_style: str = LABEL_FULL,
    region: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """Trả về bản sao của ảnh, đã vẽ khung + số thứ tự + tên + độ tin cậy.

    label_fn: hàm đổi tên loại vật trước khi vẽ (vd vi_label để hiện tiếng Việt).
    label_style: LABEL_FULL (đầy đủ), LABEL_NUMBER (chỉ số thứ tự) hoặc LABEL_NONE (chỉ khung).
    region: vùng đếm, vẽ khung vàng.
    """
    if label_style not in LABEL_STYLES:
        raise ValueError(f"Kiểu nhãn không hợp lệ: {label_style}. Chọn một trong: {', '.join(LABEL_STYLES)}")

    annotated = draw_region(image, region)  # Bản sao của ảnh (kèm vùng đếm nếu có), không sửa ảnh gốc
    draw = ImageDraw.Draw(annotated)

    # Độ dày nét và cỡ chữ tỉ lệ theo kích thước ảnh, để ảnh to hay nhỏ đều dễ nhìn
    scale = max(annotated.size) / 1000
    line_width = _line_width(annotated)
    font = find_font(max(12, round(18 * scale)))

    for index, det in enumerate(detections, start=1):
        color = PALETTE[det.class_id % len(PALETTE)]
        x1, y1, x2, y2 = det.box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        if label_style == LABEL_NONE:
            continue
        if label_style == LABEL_NUMBER:
            text = str(index)
        else:
            # Nhãn dạng "#1 người 87%": số thứ tự giúp đối chiếu khi đếm bằng mắt
            name = label_fn(det.label) if label_fn else det.label
            text = f"#{index} {name} {det.confidence:.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
        pad = max(2, line_width)
        # Đặt nhãn phía trên khung; nếu sát mép trên thì đặt vào trong khung
        label_y = y1 - text_h - 2 * pad if y1 - text_h - 2 * pad >= 0 else y1
        draw.rectangle((x1, label_y, x1 + text_w + 2 * pad, label_y + text_h + 2 * pad), fill=color)
        draw.text((x1 + pad, label_y + pad - top), text, fill=(255, 255, 255), font=font)

    return annotated
