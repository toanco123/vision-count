"""Test chế độ vật nhỏ: chia ô và gộp khung trùng."""

import pytest
from PIL import Image
from ultralytics.utils import ASSETS

from vision_count import Detection, get_detector
from vision_count.tiling import drop_cut_boxes, make_tiles, merge_overlaps


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


def test_drop_cut_boxes_removes_boxes_touching_inner_tile_edge():
    tile = (512, 0, 1152, 640)  # ô giữa: mép trái (512) là mép trong, mép trên (0) là mép ảnh
    cut = Detection("car", 2, 0.8, (512.0, 100.0, 900.0, 400.0))  # chạm mép trái bên trong: bị cắt
    at_image_top = Detection("person", 0, 0.8, (700.0, 0.0, 760.0, 100.0))  # chạm mép ảnh: giữ
    inside = Detection("person", 0, 0.8, (700.0, 200.0, 760.0, 300.0))
    assert drop_cut_boxes([cut, at_image_top, inside], tile, 1700, 640) == [at_image_top, inside]


def test_large_object_across_tiles_counted_once():
    # Xe 600px vắt qua 2 ô: mỗi ô chỉ thấy một nửa (tin cậy CAO hơn khung cả ảnh)
    full = Detection("car", 2, 0.70, (300.0, 100.0, 900.0, 400.0))
    half_a = Detection("car", 2, 0.80, (300.0, 100.0, 640.0, 400.0))
    half_b = Detection("car", 2, 0.75, (512.0, 100.0, 900.0, 400.0))
    tile_a, tile_b = (0, 0, 640, 640), (512, 0, 1152, 640)
    passes = [
        [full],
        drop_cut_boxes([half_a], tile_a, 1152, 640),
        drop_cut_boxes([half_b], tile_b, 1152, 640),
    ]
    assert merge_overlaps(passes) == [full]


def test_nested_same_class_in_one_pass_are_kept():
    # Trẻ em đứng trước người lớn: khung trẻ nằm trong khung người lớn, cùng một ô
    adult = Detection("person", 0, 0.9, (100.0, 100.0, 200.0, 400.0))
    child = Detection("person", 0, 0.8, (120.0, 250.0, 170.0, 400.0))
    assert merge_overlaps([[], [adult, child]]) == [adult, child]


def test_nested_child_seen_in_two_tiles_is_not_lost():
    adult = Detection("person", 0, 0.7, (560.0, 100.0, 630.0, 400.0))
    child_a = Detection("person", 0, 0.9, (570.0, 250.0, 610.0, 400.0))  # thấy ở ô A
    child_b = Detection("person", 0, 0.85, (571.0, 251.0, 611.0, 400.0))  # thấy lại ở ô B (vùng chồng)
    merged = merge_overlaps([[], [child_a], [adult, child_b]])
    assert merged == [child_a, adult]


def test_same_object_in_two_passes_is_merged():
    tile_box = Detection("person", 0, 0.9, (100.0, 100.0, 140.0, 200.0))
    full_box = Detection("person", 0, 0.8, (102.0, 101.0, 141.0, 199.0))
    assert merge_overlaps([[full_box], [tile_box]]) == [full_box]  # ưu tiên khung của lượt cả ảnh


def test_tile_fragment_inside_full_box_is_merged():
    # Ảnh có người rất to: một ô chỉ thấy đôi chân và tưởng là một người (tin cậy còn cao hơn)
    whole_person = Detection("person", 0, 0.6, (100.0, 100.0, 300.0, 700.0))
    legs = Detection("person", 0, 0.9, (120.0, 400.0, 280.0, 690.0))
    assert merge_overlaps([[whole_person], [legs]]) == [whole_person]


def test_merge_overlaps_keeps_different_classes():
    person = Detection("person", 0, 0.9, (0.0, 0.0, 50.0, 100.0))
    bag = Detection("handbag", 26, 0.8, (1.0, 1.0, 50.0, 99.0))  # gần trùng khung người nhưng khác loại
    assert merge_overlaps([[person], [bag]]) == [person, bag]


def test_merge_overlaps_keeps_separate_boxes():
    a = Detection("person", 0, 0.9, (0.0, 0.0, 50.0, 100.0))
    b = Detection("person", 0, 0.8, (60.0, 0.0, 110.0, 100.0))
    assert merge_overlaps([[b], [a]]) == [a, b]


@pytest.fixture(scope="module")
def tiny_people_canvas():
    """Ảnh 2400x2400 có 16 người nhỏ (cao 60px) cắt từ ảnh xe buýt mẫu.

    Chỉ dùng 3 người rõ nhất (tin cậy cao nhất): người thứ 5 trong ảnh mẫu bị khuất nửa người,
    thu nhỏ còn 60px thì model không nhận ra nữa, sẽ làm test dễ hỏng.
    """
    detector = get_detector()
    bus = Image.open(ASSETS / "bus.jpg").convert("RGB")
    people = detector.detect(bus, classes=["person"]).detections[:3]
    crops = [bus.crop(tuple(int(v) for v in d.box)) for d in people]
    canvas = Image.new("RGB", (2400, 2400), (128, 128, 128))
    for i in range(4):
        for j in range(4):
            crop = crops[(i * 4 + j) % len(crops)]
            small = crop.resize((max(1, int(crop.width * 60 / crop.height)), 60))
            canvas.paste(small, (150 + j * 560, 150 + i * 560))
    return canvas


def test_tiled_finds_small_people_normal_misses(tiny_people_canvas):
    detector = get_detector()
    normal = detector.detect(tiny_people_canvas, classes=["person"])
    tiled = detector.detect(tiny_people_canvas, classes=["person"], tiled=True)
    assert normal.total <= 3  # thu nhỏ cả ảnh về 640px thì người chỉ còn ~16px, gần như mất hết
    assert 13 <= tiled.total <= 16  # chia ô bắt được hầu hết, không đếm trùng quá số đã dán


def test_tiled_on_small_image_equals_normal():
    detector = get_detector()
    img = Image.open(ASSETS / "bus.jpg").convert("RGB").resize((600, 800))
    small = img.crop((0, 0, 600, 640))
    assert detector.detect(small, tiled=True).counts == detector.detect(small).counts
