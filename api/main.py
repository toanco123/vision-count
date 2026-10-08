"""API FastAPI cho vision-count. Chạy: uvicorn api.main:app --port 8000, rồi mở http://127.0.0.1:8000/docs

Chỉ gọi tới package vision_count, giống hệt giao diện Gradio, nên app React/Flutter sau này
dùng chung đúng một phần AI.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

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


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _parse_classes(classes: str) -> list[str] | None:
    names = [c.strip() for c in classes.split(",") if c.strip()]
    return names or None


def _parse_region(region: str):
    if not region.strip():
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in region.split(","))
    except ValueError:
        raise HTTPException(status_code=400, detail="Vùng phải có dạng 'x1,y1,x2,y2', ví dụ '0,0,400,600'.")
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _detect(file: UploadFile, model: str, confidence: float, classes: str, tiled: bool, region: str):
    """Phần dùng chung của /detect và /detect/image: đọc ảnh, đếm, lọc theo vùng."""
    region_box = _parse_region(region)
    try:
        detector = get_detector(model)
        image = load_image(file.file.read(), name=file.filename)
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
    try:
        names = get_detector(model).class_names
    except ValueError as exc:
        raise _bad_request(exc) from exc
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
        with open(path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        try:
            # count_video dùng bản model riêng nên không cần khóa
            result = count_video(
                get_detector(model),
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
