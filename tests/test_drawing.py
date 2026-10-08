"""Test phần vẽ khung và nhãn."""

from PIL import Image, ImageFont

from vision_count import Detection, draw_detections, vi_label
from vision_count.drawing import find_font


def _sample_detection():
    return Detection(label="person", class_id=0, confidence=0.9, box=(10, 30, 120, 180))


def test_find_font_returns_truetype_font_on_this_machine():
    # Máy dev (macOS) có Arial hỗ trợ tiếng Việt; font mặc định của Pillow thì không
    assert isinstance(find_font(20), ImageFont.FreeTypeFont)


def test_label_fn_is_used_for_text():
    seen = []

    def spy(label):
        seen.append(label)
        return vi_label(label)

    draw_detections(Image.new("RGB", (200, 200)), [_sample_detection()], label_fn=spy)
    assert seen == ["person"]


def test_default_draws_without_label_fn():
    out = draw_detections(Image.new("RGB", (200, 200)), [_sample_detection()])
    assert out.size == (200, 200)
