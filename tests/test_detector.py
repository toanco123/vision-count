"""Test phần xử lý AI. Chạy: pytest"""

import io

import pytest
from PIL import Image
from ultralytics.utils import ASSETS  # Thư mục ảnh mẫu có sẵn trong thư viện Ultralytics

from vision_count import InvalidImageError, ObjectDetector, draw_detections, load_image

BUS_IMAGE = ASSETS / "bus.jpg"  # Ảnh xe buýt có vài người đứng trước


@pytest.fixture(scope="module")
def detector():
    # Nạp model một lần cho cả file test (nạp model khá chậm)
    return ObjectDetector()


# ---------- load_image: kiểm tra đầu vào ----------

def test_load_image_rejects_text_file(tmp_path):
    fake = tmp_path / "ghi_chu.jpg"  # Đuôi .jpg nhưng nội dung là chữ
    fake.write_text("day khong phai anh")
    with pytest.raises(InvalidImageError):
        load_image(fake)


def test_load_image_rejects_random_bytes():
    with pytest.raises(InvalidImageError):
        load_image(b"\x00\x01\x02 khong phai anh")


def test_load_image_rejects_missing_file():
    with pytest.raises(InvalidImageError):
        load_image("khong_ton_tai.png")


def test_load_image_accepts_png_bytes_and_converts_to_rgb():
    buffer = io.BytesIO()
    Image.new("RGBA", (50, 40), (255, 0, 0, 128)).save(buffer, format="PNG")
    img = load_image(buffer.getvalue())
    assert img.mode == "RGB"
    assert img.size == (50, 40)


# ---------- ObjectDetector ----------

def test_has_80_coco_classes(detector):
    assert len(detector.class_names) == 80
    assert "person" in detector.class_names


def test_blank_image_has_no_objects(detector):
    result = detector.detect(Image.new("RGB", (640, 480), "white"))
    assert result.total == 0
    assert result.counts == {}


def test_bus_image_detects_people_and_bus(detector):
    result = detector.detect(BUS_IMAGE)
    assert result.counts.get("person", 0) >= 3
    assert result.counts.get("bus", 0) >= 1
    assert result.total == sum(result.counts.values())
    assert all(d.confidence >= 0.25 for d in result.detections)


def test_class_filter_only_counts_selected(detector):
    result = detector.detect(BUS_IMAGE, classes=["bus"])
    assert set(result.counts) == {"bus"}


def test_higher_confidence_gives_fewer_or_equal_boxes(detector):
    low = detector.detect(BUS_IMAGE, confidence=0.1)
    high = detector.detect(BUS_IMAGE, confidence=0.8)
    assert high.total <= low.total


def test_unknown_class_raises(detector):
    with pytest.raises(ValueError):
        detector.detect(BUS_IMAGE, classes=["khung_long"])


def test_invalid_confidence_raises(detector):
    with pytest.raises(ValueError):
        detector.detect(BUS_IMAGE, confidence=1.5)


def test_to_dict_is_json_ready(detector):
    import json

    data = detector.detect(BUS_IMAGE).to_dict()
    json.dumps(data)  # Không lỗi = trả về JSON được (dùng cho FastAPI sau này)
    assert data["total"] == len(data["detections"])


def test_draw_detections_keeps_size_and_original(detector):
    original = load_image(BUS_IMAGE)
    result = detector.detect(original)
    annotated = draw_detections(original, result.detections)
    assert annotated.size == original.size
    assert annotated is not original


def test_load_image_error_uses_given_name():
    with pytest.raises(InvalidImageError, match="anh_cua_toi.png"):
        load_image(b"khong phai anh", name="anh_cua_toi.png")
