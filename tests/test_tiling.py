"""Test chế độ vật nhỏ: chia ô và gộp khung trùng."""

import pytest
from PIL import Image
from ultralytics.utils import ASSETS

from vision_count import Detection, get_detector
from vision_count.tiling import make_tiles, merge_overlaps


def test_small_image_is_one_tile():
    assert make_tiles(500, 400) == [(0, 0, 500, 400)]


def test_tiles_cover_whole_image_without_exceeding_size():
    tiles = make_tiles(2000, 1300, tile=640, overlap=0.2)
    assert all(x2 - x1 <= 640 and y2 - y1 <= 640 for x1, y1, x2, y2 in tiles)
    assert max(x2 for _, _, x2, _ in tiles) == 2000
    assert max(y2 for _, _, _, y2 in tiles) == 1300
    # Mọi điểm trên lưới 10px của ảnh (kể cả sát mép phải/dưới) đều nằm trong ít nhất một ô
    points = [(x, y) for x in [*range(0, 2000, 10), 1999] for y in [*range(0, 1300, 10), 1299]]
    assert all(any(x1 <= x < x2 and y1 <= y < y2 for x1, y1, x2, y2 in tiles) for x, y in points)


def test_merge_overlaps_merges_partial_box_at_tile_edge():
    full = Detection("person", 0, 0.9, (100.0, 100.0, 140.0, 200.0))
    half = Detection("person", 0, 0.6, (100.0, 100.0, 140.0, 150.0))  # nửa trên, nằm trong khung đầy đủ
    assert merge_overlaps([half, full]) == [full]


def test_merge_overlaps_keeps_different_classes():
    person = Detection("person", 0, 0.9, (0.0, 0.0, 50.0, 100.0))
    bag = Detection("handbag", 26, 0.8, (10.0, 40.0, 30.0, 60.0))  # nằm trong người nhưng khác loại
    assert merge_overlaps([person, bag]) == [person, bag]


def test_merge_overlaps_keeps_separate_boxes():
    a = Detection("person", 0, 0.9, (0.0, 0.0, 50.0, 100.0))
    b = Detection("person", 0, 0.8, (60.0, 0.0, 110.0, 100.0))
    assert merge_overlaps([b, a]) == [a, b]


@pytest.fixture(scope="module")
def tiny_people_canvas():
    """Ảnh 2400x2400 có 16 người nhỏ (cao 80px) cắt từ ảnh xe buýt mẫu."""
    detector = get_detector()
    bus = Image.open(ASSETS / "bus.jpg").convert("RGB")
    people = detector.detect(bus, classes=["person"]).detections[:4]
    crops = [bus.crop(tuple(int(v) for v in d.box)) for d in people]
    canvas = Image.new("RGB", (2400, 2400), (128, 128, 128))
    for i in range(4):
        for j in range(4):
            crop = crops[(i * 4 + j) % len(crops)]
            small = crop.resize((max(1, int(crop.width * 80 / crop.height)), 80))
            canvas.paste(small, (150 + j * 560, 150 + i * 560))
    return canvas


def test_tiled_finds_small_people_normal_misses(tiny_people_canvas):
    detector = get_detector()
    normal = detector.detect(tiny_people_canvas, classes=["person"])
    tiled = detector.detect(tiny_people_canvas, classes=["person"], tiled=True)
    assert normal.total <= 2  # thu nhỏ cả ảnh về 640px thì người chỉ còn ~20px, gần như mất hết
    assert 12 <= tiled.total <= 16  # chia ô bắt được hầu hết, không đếm trùng quá số đã dán


def test_tiled_on_small_image_equals_normal():
    detector = get_detector()
    img = Image.open(ASSETS / "bus.jpg").convert("RGB").resize((600, 800))
    small = img.crop((0, 0, 600, 640))
    assert detector.detect(small, tiled=True).counts == detector.detect(small).counts
