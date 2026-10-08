"""Test vùng đếm."""

from PIL import Image, ImageChops

from vision_count import (
    Detection,
    DetectionResult,
    draw_detections,
    draw_region,
    filter_by_region,
    normalize_region,
)
from vision_count.drawing import REGION_COLOR


def _result():
    dets = [
        Detection("person", 0, 0.9, (0.0, 0.0, 20.0, 20.0)),  # tâm (10, 10): trong vùng
        Detection("person", 0, 0.8, (80.0, 80.0, 100.0, 100.0)),  # tâm (90, 90): ngoài vùng
        Detection("car", 2, 0.7, (30.0, 30.0, 70.0, 50.0)),  # tâm (50, 40): trong vùng
    ]
    return DetectionResult(detections=dets, counts={"person": 2, "car": 1}, image_width=100, image_height=100)


def test_normalize_region_orders_corners():
    assert normalize_region((60, 70), (5, 10)) == (5, 10, 60, 70)


def test_filter_keeps_only_centers_inside_and_recounts():
    filtered = filter_by_region(_result(), (0, 0, 60, 60))
    assert [d.confidence for d in filtered.detections] == [0.9, 0.7]
    assert filtered.counts == {"person": 1, "car": 1}
    assert filtered.total == 2
    assert (filtered.image_width, filtered.image_height) == (100, 100)


def test_filter_center_on_edge_counts():
    filtered = filter_by_region(_result(), (10, 10, 50, 40))  # đúng bằng tâm 2 khung
    assert filtered.total == 2


def test_filter_without_region_returns_same():
    result = _result()
    assert filter_by_region(result, None) is result


def test_draw_detections_shows_region():
    img = Image.new("RGB", (100, 100))
    with_region = draw_detections(img, [], region=(10, 10, 60, 60))
    assert with_region.getpixel((10, 30)) == REGION_COLOR
    assert ImageChops.difference(with_region, img).getbbox() is not None


def test_draw_region_marks_point_and_keeps_original():
    img = Image.new("RGB", (100, 100))
    marked = draw_region(img, point=(50, 50))
    assert marked.getpixel((50, 50)) == REGION_COLOR
    assert img.getpixel((50, 50)) == (0, 0, 0)
