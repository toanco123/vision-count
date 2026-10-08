"""Test API FastAPI bằng TestClient (không cần chạy server thật)."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from ultralytics.utils import ASSETS

from api.main import app

client = TestClient(app)
BUS = ASSETS / "bus.jpg"


def _post_image(url="/detect", path=BUS, **form):
    with open(path, "rb") as f:
        return client.post(url, files={"file": (path.name, f, "image/jpeg")}, data=form)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_models_lists_nano_and_small():
    keys = [m["key"] for m in client.get("/models").json()]
    assert keys[:2] == ["nano", "small"]


def test_classes_have_vietnamese_names():
    classes = client.get("/classes", params={"model": "nano"}).json()
    assert {"label": "person", "label_vi": "người"} in classes and len(classes) == 80


def test_detect_returns_counts_and_boxes():
    data = _post_image().json()
    assert data["model"] == "nano"
    assert data["total"] == 5 and data["counts"] == {"person": 4, "bus": 1}
    assert data["counts_vi"] == {"người": 4, "xe buýt": 1}
    assert data["detections"][0]["label_vi"] in {"người", "xe buýt"}
    assert (data["image_width"], data["image_height"]) == (810, 1080)


def test_detect_with_class_filter_and_region():
    assert _post_image(classes="person").json()["counts"] == {"person": 4}
    data = _post_image(region="0,0,400,1080").json()
    assert data["region"] == [0, 0, 400, 1080] and data["total"] < 5


def test_detect_image_returns_jpeg():
    resp = _post_image("/detect/image", label_style="number")
    assert resp.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(resp.content)).size == (810, 1080)


def test_api_bad_image_400(tmp_path):
    bad = tmp_path / "hong.jpg"
    bad.write_text("khong phai anh")
    resp = _post_image(path=bad)
    assert resp.status_code == 400 and "không phải là file ảnh" in resp.json()["detail"]
    assert "hong.jpg" in resp.json()["detail"]  # nói rõ file nào bị lỗi


@pytest.mark.parametrize(
    "form, text",
    [({"model": "khong_co"}, "Không có model"), ({"classes": "khung_long"}, "không biết"), ({"region": "1,2,3"}, "Vùng")],
)
def test_api_bad_params_400(form, text):
    resp = _post_image(**form)
    assert resp.status_code == 400 and text in resp.json()["detail"]


def test_api_confidence_out_of_range_422():
    assert _post_image(confidence="2").status_code == 422


def test_api_video_count(synthetic_video):
    with open(synthetic_video, "rb") as f:
        data = client.post("/video/count", files={"file": ("synthetic.mp4", f, "video/mp4")}).json()
    assert data["counts"] == {"person": 3} and data["counts_vi"] == {"người": 3}
    assert data["frames_processed"] == 30
    assert data["complete"] is True and data["warning"] is None


@pytest.mark.parametrize("filename", ["blob", "clip.3gp", "..", "a" * 300 + ".mp4"])
def test_api_video_accepts_any_upload_name(synthetic_video, filename):
    # Trình duyệt gửi Blob với tên "blob"; OpenCV nhận dạng video theo nội dung nên tên file không quan trọng
    with open(synthetic_video, "rb") as f:
        resp = client.post("/video/count", files={"file": (filename, f, "video/mp4")})
    assert resp.status_code == 200 and resp.json()["counts"] == {"person": 3}


def test_api_video_rejects_image():
    with open(BUS, "rb") as f:
        resp = client.post("/video/count", files={"file": ("bus.jpg", f, "image/jpeg")})
    assert resp.status_code == 400
    assert "ảnh" in resp.json()["detail"] and "bus.jpg" in resp.json()["detail"]


def test_api_video_rejects_non_video(tmp_path):
    bad = tmp_path / "hong.mp4"
    bad.write_text("x")
    with open(bad, "rb") as f:
        resp = client.post("/video/count", files={"file": ("hong.mp4", f, "video/mp4")})
    assert resp.status_code == 400 and "không phải là file video" in resp.json()["detail"]


# ---------- Đợt sửa điểm nhỏ ----------

def test_api_validation_errors_are_vietnamese():
    resp = _post_image(confidence="2")
    assert resp.status_code == 422 and "Ngưỡng độ tin cậy" in resp.json()["detail"]
    resp = client.post("/detect", data={"model": "nano"})  # thiếu file
    assert resp.status_code == 422 and "file" in resp.json()["detail"] and "Thiếu" in resp.json()["detail"]


@pytest.mark.parametrize("region", ["nan,nan,nan,nan", "0,0,inf,inf"])
def test_api_region_must_be_finite(region):
    resp = _post_image(region=region)
    assert resp.status_code == 400 and "Vùng" in resp.json()["detail"]


def test_api_rejects_too_large_upload(monkeypatch):
    import api.main as api_main

    monkeypatch.setattr(api_main, "MAX_UPLOAD_MB", 0.01)  # ~10KB, ảnh mẫu 134KB
    resp = _post_image()
    assert resp.status_code == 413 and "quá lớn" in resp.json()["detail"]


def test_api_model_load_failure_is_503(monkeypatch):
    import api.main as api_main

    def offline(_key):
        raise RuntimeError("Download failure, environment may be offline")

    monkeypatch.setattr(api_main, "get_detector", offline)
    resp = _post_image()
    assert resp.status_code == 503 and "Không nạp được model" in resp.json()["detail"]
