"""Vẽ khung và nhãn lên ảnh. Tách riêng để giao diện nào cũng dùng được."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from vision_count.detector import Detection

# Bảng màu: mỗi loại vật (class_id) lấy một màu cố định để dễ phân biệt
PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
    (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
    (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Trả về bản sao của ảnh, đã vẽ khung + số thứ tự + tên + độ tin cậy."""
    annotated = image.convert("RGB").copy()  # Không sửa ảnh gốc
    draw = ImageDraw.Draw(annotated)

    # Độ dày nét và cỡ chữ tỉ lệ theo kích thước ảnh, để ảnh to hay nhỏ đều dễ nhìn
    scale = max(annotated.size) / 1000
    line_width = max(2, round(3 * scale))
    font = ImageFont.load_default(size=max(12, round(18 * scale)))

    for index, det in enumerate(detections, start=1):
        color = PALETTE[det.class_id % len(PALETTE)]
        x1, y1, x2, y2 = det.box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        # Nhãn dạng "#1 person 87%": số thứ tự giúp đối chiếu khi đếm bằng mắt
        text = f"#{index} {det.label} {det.confidence:.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
        pad = max(2, line_width)
        # Đặt nhãn phía trên khung; nếu sát mép trên thì đặt vào trong khung
        label_y = y1 - text_h - 2 * pad if y1 - text_h - 2 * pad >= 0 else y1
        draw.rectangle((x1, label_y, x1 + text_w + 2 * pad, label_y + text_h + 2 * pad), fill=color)
        draw.text((x1 + pad, label_y + pad - top), text, fill=(255, 255, 255), font=font)

    return annotated
