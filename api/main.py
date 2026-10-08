"""API FastAPI cho vision-count. Chạy: uvicorn api.main:app --port 8000, rồi mở http://127.0.0.1:8000/docs

Chỉ gọi tới package vision_count, giống hệt giao diện Gradio, nên app React/Flutter sau này
dùng chung đúng một phần AI.
"""

from __future__ import annotations

import io
import math
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from vision_count import (
    LABEL_FULL,
    InvalidImageError,
    InvalidVideoError,
    config,
    count_video,
    draw_detections,
    filter_by_region,
    get_detector,
    is_model_downloaded,
    load_image,
    vi_label,
)

app = FastAPI(
    title="vision-count API",
    description="Đếm vật thể trong ảnh và video bằng YOLO. Mọi thứ chạy trên máy, không gọi dịch vụ ngoài.",
    version="1.0",
)

# FastAPI chạy các hàm `def` trên nhiều luồng; model YOLO dùng chung không an toàn khi 2 luồng gọi cùng lúc
_model_lock = threading.Lock()

# Giới hạn dung lượng file tải lên, tránh một file khổng lồ làm đầy bộ nhớ/ổ đĩa
MAX_UPLOAD_MB = 50
MAX_VIDEO_MB = 2000

_FIELD_MESSAGES = {
    "confidence": "Ngưỡng độ tin cậy (confidence) phải là số từ 0 đến 1.",
    "vid_stride": "vid_stride phải là số nguyên từ 1 đến 10.",
    "tiled": "tiled phải là true hoặc false.",
}


@app.exception_handler(RequestValidationError)
def _validation_error(request, exc: RequestValidationError):
    """Lỗi kiểm tra tham số (422) bằng tiếng Việt thay vì thông báo tiếng Anh mặc định."""
    messages = []
    for err in exc.errors():
        field = str(err["loc"][-1]) if err.get("loc") else "?"
        if err.get("type") == "missing":
            messages.append(f"Thiếu trường bắt buộc '{field}'" + (" (file cần đếm)." if field == "file" else "."))
        else:
            messages.append(_FIELD_MESSAGES.get(field, f"Giá trị của '{field}' không hợp lệ."))
    return JSONResponse(status_code=422, content={"detail": " ".join(dict.fromkeys(messages))})


def _too_large(limit_mb: float) -> HTTPException:
    return HTTPException(status_code=413, detail=f"File quá lớn (tối đa {limit_mb:g}MB).")


def _get_model(model: str):
    """Nạp model: tên sai → 400; tải lần đầu thất bại (mất mạng...) → 503."""
    try:
        return get_detector(model)
    except ValueError as exc:
        raise _bad_request(exc) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Không nạp được model '{model}': {exc}. Lần đầu dùng model cần mạng để tải.",
        ) from exc


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _parse_classes(classes: str) -> list[str] | None:
    names = [c.strip() for c in classes.split(",") if c.strip()]
    return names or None


def _parse_region(region: str):
    if not region.strip():
        return None
    message = "Vùng phải có dạng 'x1,y1,x2,y2' (4 số), ví dụ '0,0,400,600'."
    try:
        x1, y1, x2, y2 = (float(v) for v in region.split(","))
    except ValueError:
        raise HTTPException(status_code=400, detail=message)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):  # chặn nan, inf
        raise HTTPException(status_code=400, detail=message)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _detect(file: UploadFile, model: str, confidence: float, classes: str, tiled: bool, region: str):
    """Phần dùng chung của /detect và /detect/image: đọc ảnh, đếm, lọc theo vùng."""
    region_box = _parse_region(region)
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise _too_large(MAX_UPLOAD_MB)
    detector = _get_model(model)
    try:
        image = load_image(data, name=file.filename)
        with _model_lock:
            result = detector.detect(image, confidence=confidence, classes=_parse_classes(classes), tiled=tiled)
    except (InvalidImageError, ValueError) as exc:
        raise _bad_request(exc) from exc
    return image, filter_by_region(result, region_box), region_box


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    """Các model chọn được (giống ô Model trên giao diện)."""
    return [
        {"key": key, "name": name, "downloaded": is_model_downloaded(key)}
        for key, (name, _) in config.AVAILABLE_MODELS.items()
    ]


@app.get("/classes")
def classes(model: str = config.DEFAULT_MODEL_KEY):
    """Các loại vật model nhận được, kèm tên tiếng Việt."""
    names = _get_model(model).class_names
    return [{"label": n, "label_vi": vi_label(n)} for n in names]


@app.post("/detect")
def detect(
    file: UploadFile = File(..., description="File ảnh (JPG, PNG, WEBP...)"),
    model: str = Form(config.DEFAULT_MODEL_KEY),
    confidence: float = Form(config.DEFAULT_CONFIDENCE, ge=0, le=1),
    classes: str = Form("", description="Chỉ đếm các loại này, ngăn cách bằng dấu phẩy, vd 'person,car'"),
    tiled: bool = Form(False, description="Chế độ vật nhỏ (chia ô)"),
    region: str = Form("", description="Vùng đếm 'x1,y1,x2,y2' theo pixel; để trống = cả ảnh"),
):
    """Đếm vật thể trong ảnh, trả về JSON."""
    _, result, region_box = _detect(file, model, confidence, classes, tiled, region)
    data = result.to_dict()
    for d in data["detections"]:
        d["label_vi"] = vi_label(d["label"])
    return {
        "model": model,
        "total": data["total"],
        "counts": data["counts"],
        "counts_vi": {vi_label(label): n for label, n in data["counts"].items()},
        "detections": data["detections"],
        "image_width": data["image_width"],
        "image_height": data["image_height"],
        "region": list(region_box) if region_box else None,
    }


@app.post("/detect/image", response_class=Response, responses={200: {"content": {"image/jpeg": {}}}})
def detect_image(
    file: UploadFile = File(...),
    model: str = Form(config.DEFAULT_MODEL_KEY),
    confidence: float = Form(config.DEFAULT_CONFIDENCE, ge=0, le=1),
    classes: str = Form(""),
    tiled: bool = Form(False),
    region: str = Form(""),
    label_style: str = Form(LABEL_FULL, description="full | number | none"),
):
    """Đếm vật thể và trả về ảnh JPEG đã khoanh khung."""
    image, result, region_box = _detect(file, model, confidence, classes, tiled, region)
    try:
        annotated = draw_detections(
            image, result.detections, label_fn=vi_label, label_style=label_style, region=region_box
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc
    buffer = io.BytesIO()
    annotated.save(buffer, format="JPEG", quality=90)
    return Response(content=buffer.getvalue(), media_type="image/jpeg")


@app.post("/video/count")
def video_count(
    file: UploadFile = File(..., description="File video (MP4, MOV, AVI...)"),
    model: str = Form(config.DEFAULT_MODEL_KEY),
    confidence: float = Form(config.DEFAULT_CONFIDENCE, ge=0, le=1),
    classes: str = Form(""),
    vid_stride: int = Form(1, ge=1, le=10, description="Chỉ xử lý 1 trên N khung hình"),
):
    """Đếm số vật KHÁC NHAU trong video (có tracking). Video dài sẽ mất vài phút."""
    from ultralytics.data.utils import VID_FORMATS

    original = file.filename or "video"
    suffix = Path(original).suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        # Không dùng tên file của người gửi (có thể là "blob", "..", quá dài...). OpenCV nhận dạng video
        # theo nội dung, nên đuôi lạ (vd .3gp) cứ lưu thành .mp4; chỉ giữ đuôi mà Ultralytics biết.
        path = Path(tmp) / ("video" + (suffix if suffix.lstrip(".") in VID_FORMATS else ".mp4"))
        limit, written = int(MAX_VIDEO_MB * 1024 * 1024), 0
        with open(path, "wb") as out:
            while chunk := file.file.read(1024 * 1024):  # chép từng 1MB, dừng nếu vượt giới hạn
                written += len(chunk)
                if written > limit:
                    raise _too_large(MAX_VIDEO_MB)
                out.write(chunk)
        detector = _get_model(model)
        try:
            # count_video dùng bản model riêng nên không cần khóa
            result = count_video(
                detector,
                path,
                confidence=confidence,
                classes=_parse_classes(classes),
                vid_stride=vid_stride,
                name=original,
            )
        except (InvalidVideoError, ValueError) as exc:
            raise _bad_request(exc) from exc
    return {
        "model": model,
        "total": result.total,
        "counts": result.counts,
        "counts_vi": {vi_label(label): n for label, n in result.counts.items()},
        "frames_processed": result.frames_processed,
        "fps": result.fps,
        "duration_s": result.duration_s,
        "peak_in_frame": result.peak_in_frame,
        "complete": result.complete,  # False: video hỏng giữa chừng, chỉ đếm được phần đầu
        "warning": result.warning,
    }
