"""Test phần vẽ khung và nhãn."""

import pytest
from PIL import Image, ImageChops, ImageDraw

from vision_count import LABEL_FULL, LABEL_NONE, LABEL_NUMBER, Detection, draw_detections, vi_label
from vision_count import drawing
from vision_count.drawing import find_font


def _sample_detection():
    return Detection(label="person", class_id=0, confidence=0.9, box=(10, 30, 120, 180))


def _draw(style, label_fn=None):
    return draw_detections(Image.new("RGB", (200, 200)), [_sample_detection()], label_fn=label_fn, label_style=style)


@pytest.fixture
def clear_font_cache():
    drawing._find_font_name.cache_clear()
    find_font.cache_clear()
    yield
    drawing._find_font_name.cache_clear()
    find_font.cache_clear()


@pytest.mark.skipif(drawing._find_font_name() is None, reason="Máy không có font hỗ trợ tiếng Việt")
def test_find_font_uses_a_real_font_file():
    # Font thật có đường dẫn file (str); font mặc định của Pillow nằm trong bộ nhớ (không có dấu tiếng Việt)
    assert isinstance(find_font(20).path, str)


def test_find_font_falls_back_without_vietnamese_font(monkeypatch, clear_font_cache):
    monkeypatch.setattr(drawing, "VIETNAMESE_FONTS", ["khong_co_font_nay.ttf"])
    font = find_font(20)
    assert not isinstance(font.path, str)  # đã dùng font mặc định
    _draw(LABEL_FULL, vi_label)  # vẫn vẽ được, không lỗi


def test_label_fn_is_used_for_full_label():
    seen = []
    _draw(LABEL_FULL, lambda label: seen.append(label) or vi_label(label))
    assert seen == ["person"]


def test_number_style_does_not_need_name():
    seen = []
    _draw(LABEL_NUMBER, lambda label: seen.append(label) or label)
    assert seen == []


def test_styles_draw_different_images():
    full, number, none = _draw(LABEL_FULL), _draw(LABEL_NUMBER), _draw(LABEL_NONE)
    assert ImageChops.difference(full, number).getbbox() is not None
    assert ImageChops.difference(number, none).getbbox() is not None


def test_none_style_draws_only_box():
    # Ảnh 200px: độ dày nét = max(2, round(3 * 0.2)) = 2; class_id 0 dùng màu PALETTE[0]
    expected = Image.new("RGB", (200, 200))
    ImageDraw.Draw(expected).rectangle((10, 30, 120, 180), outline=drawing.PALETTE[0], width=2)
    assert ImageChops.difference(_draw(LABEL_NONE), expected).getbbox() is None


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        _draw("khong_co")


def test_default_draws_without_label_fn():
    out = draw_detections(Image.new("RGB", (200, 200)), [_sample_detection()])
    assert out.size == (200, 200)
