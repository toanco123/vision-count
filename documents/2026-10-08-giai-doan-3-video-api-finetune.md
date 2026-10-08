# Giai đoạn 3: Đếm video có tracking, API FastAPI, bộ công cụ fine-tune: Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Mục tiêu:**
1. **Đếm video có tracking:** tải lên một file video, đếm số vật **khác nhau** đã xuất hiện. Mỗi vật có một ID (ByteTrack) nên không đếm trùng. Kết quả có video đã vẽ khung kèm ID và file CSV.
2. **API FastAPI:** bọc phần AI thành API (`/detect`, `/detect/image`, `/video/count`, `/models`, `/classes`), để sau này React hoặc Flutter gọi vào.
3. **Bộ công cụ fine-tune:** người dùng chưa có ảnh riêng, nên chuẩn bị sẵn:
   - hướng dẫn thu thập ảnh và gán nhãn;
   - mẫu `data.yaml`;
   - notebook Google Colab;
   - script kiểm tra bộ dữ liệu;
   - ô "Chỉ đếm các loại" tự cập nhật theo model đang chọn (để model fine-tune hiện đúng các loại của nó).

**Kiến trúc:**
- **Lõi video:** `vision_count/video.py` dùng `model.track(...)` của Ultralytics. Mỗi lần đếm tạo **một bản YOLO riêng** từ `detector.model_path`, để trạng thái tracking không dính vào model dùng chung và an toàn khi chạy song song với đếm ảnh.
- **API:** `api/main.py` chỉ gọi `vision_count`, giống hệt Gradio. Có khóa (`threading.Lock`) quanh phần dùng model, vì FastAPI chạy các request đồng bộ trên nhiều luồng.
- **Fine-tune:** thư mục `training/` độc lập, không ảnh hưởng ứng dụng.

**Công nghệ:** Ultralytics `model.track` + ByteTrack (cần thư viện `lap`), OpenCV `VideoWriter` codec `avc1` (H.264, trình duyệt phát được, không cần ffmpeg), FastAPI + uvicorn + python-multipart (đã có sẵn vì Gradio dùng), PyYAML (Ultralytics đã kèm), pytest + `fastapi.testclient`.

**Spec:** Không có file spec riêng. Yêu cầu là nhóm 3 (mục 8-10) trong danh sách đề xuất ngày 2026-10-08. Người dùng chọn: video theo cách (a), đếm số vật khác nhau có ID; fine-tune thì chỉ chuẩn bị công cụ vì chưa có ảnh.

## Ràng buộc chung

- Chạy offline sau khi cài đặt.
- Thêm vào `requirements.txt`: `fastapi`, `uvicorn`, `python-multipart` (đã cài sẵn theo Gradio, ghi ra cho rõ), và `lap` (ByteTrack cần; không ghi thì Ultralytics tự `pip install` lúc chạy, tức là cần mạng giữa chừng).
- Không đổi hành vi của `detect()` và giao diện hiện có.
- Chú thích code bằng tiếng Việt.

## Điểm đã kiểm chứng trước khi viết kế hoạch

- Video tự tạo có 3 người đi ngang trong 30 khung hình. ByteTrack cho đúng **3 ID**, trong khi cộng số đếm từng khung lại ra **79**.
- OpenCV trên máy ghi được `avc1`: đọc lại thấy fourcc là `h264`.
- `gr.Video` chỉ cần ffmpeg khi đặt `format` hoặc lật hình webcam. Chỉ cho tải file lên + video kết quả đã là H.264 thì không cần ffmpeg.
- `gr.Progress()` gọi ngoài sự kiện Gradio (trong test) không lỗi.
- OpenCV mở file hỏng thì `isOpened()` trả về False.

## Điểm cần chú ý khi review

1. **File không phải video** (hoặc video hỏng): phải báo lỗi tiếng Việt, không treo. Test: `test_invalid_video_raises`, `test_api_video_rejects_non_video`.
2. **Một vật bị nhận diện chập chờn** (chỉ xuất hiện 1-2 khung): không được tính là vật mới. Mỗi ID phải xuất hiện ít nhất `MIN_TRACK_FRAMES = 3` khung mới được đếm. Test: `test_short_tracks_are_ignored`.
3. **Đếm video xong rồi đếm ảnh:** đếm ảnh không được bị ảnh hưởng. Test: `test_image_detection_unchanged_after_video`.
4. **API nhận ảnh hỏng, tên model sai, tên loại sai, vùng sai định dạng:** trả về 400 kèm thông báo tiếng Việt; ngưỡng ngoài 0..1 trả 422. Test: các `test_api_*_400`.
5. **Đổi sang model fine-tune có loại riêng:** ô lọc phải đổi theo. Test: `test_class_choices_follow_model`.

---

## Cấu trúc file

| File | Việc | Trách nhiệm |
|---|---|---|
| `vision_count/video.py` | Tạo | `InvalidVideoError`, `VideoCountResult`, `count_video`, `write_video_csv` |
| `vision_count/drawing.py` | Sửa | Tách hàm `draw_label` để vẽ nhãn dùng chung cho ảnh và video |
| `vision_count/history.py` | Sửa | `add()` nhận mọi kết quả có `.total`/`.counts` (cả video) |
| `vision_count/config.py` | Sửa | `MIN_TRACK_FRAMES`, `TRACKER` |
| `vision_count/__init__.py` | Sửa | Xuất tên mới |
| `tests/conftest.py` | Tạo | Fixture video tự tạo dùng chung |
| `ui/handlers.py`, `ui/gradio_app.py` | Sửa | Tab Video; ô lọc đổi theo model |
| `api/__init__.py`, `api/main.py` | Tạo | FastAPI |
| `training/README.md`, `training/data.yaml`, `training/train_colab.ipynb`, `training/check_dataset.py` | Tạo | Bộ công cụ fine-tune |
| `tests/test_video.py`, `tests/test_api.py`, `tests/test_check_dataset.py` | Tạo | |
| `requirements.txt`, `README.md` | Sửa | |

---

### Task 1: Lõi đếm video (`vision_count/video.py`)

**Files:** Tạo `vision_count/video.py`, `tests/conftest.py`, `tests/test_video.py`. Sửa `vision_count/drawing.py`, `vision_count/history.py`, `vision_count/config.py`, `vision_count/__init__.py`, `requirements.txt`.

**Interfaces:**
- Produces:
  - `InvalidVideoError(ValueError)`.
  - `VideoCountResult(counts: dict[str, int], frames_processed: int, fps: float, duration_s: float, peak_in_frame: int, output_path: Path | None)`, có thuộc tính `.total`.
  - `count_video(detector, video_path, confidence=0.25, classes=None, vid_stride=1, output_path=None, label_fn=None, progress=None, min_track_frames=config.MIN_TRACK_FRAMES) -> VideoCountResult`. Trong đó `progress(done: int, total: int)`.
  - `write_video_csv(result, path) -> Path`.
  - `draw_label(draw, x, y, text, color, font, pad)` (trong drawing.py).
  - Fixture `synthetic_video` (session) → `Path` tới mp4 30 khung, 15 fps, 640×480, 3 người đi ngang.

- [ ] **Bước 1: Viết test**

`tests/conftest.py`:

```python
"""Fixture dùng chung cho nhiều file test."""

import cv2
import numpy as np
import pytest
from PIL import Image
from ultralytics.utils import ASSETS


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """Video 30 khung, 15 fps, 640x480: 3 người (cắt từ ảnh xe buýt mẫu) đi ngang với tốc độ khác nhau."""
    from vision_count import get_detector

    bus = Image.open(ASSETS / "bus.jpg").convert("RGB")
    people = get_detector().detect(bus, classes=["person"]).detections[:3]
    crops = [bus.crop(tuple(int(v) for v in d.box)) for d in people]
    crops = [c.resize((int(c.width * 200 / c.height), 200)) for c in crops]

    path = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    width, height, n_frames = 640, 480, 30
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), 15, (width, height))
    for f in range(n_frames):
        frame = Image.new("RGB", (width, height), (110, 120, 110))
        for k, crop in enumerate(crops):
            # Di chuyển liên tục (không "dịch chuyển tức thời"), mỗi người một làn và một tốc độ
            x = int(20 + (width - 160) * f / (n_frames - 1) * (0.6 + 0.2 * k))
            frame.paste(crop, (x, 20 + k * 130))
        writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    writer.release()
    return path
```

`tests/test_video.py`:

```python
"""Test đếm video có tracking."""

import csv

import cv2
import pytest
from ultralytics.utils import ASSETS

from vision_count import InvalidVideoError, VideoCountResult, count_video, get_detector, vi_label, write_video_csv


def test_counts_each_person_once(synthetic_video):
    result = count_video(get_detector(), synthetic_video)
    assert result.counts == {"person": 3}  # cộng từng khung sẽ ra ~79
    assert result.total == 3
    assert result.frames_processed == 30
    assert result.fps == 15
    assert result.duration_s == 2
    assert result.peak_in_frame == 3
    assert result.output_path is None


def test_writes_playable_annotated_video(synthetic_video, tmp_path):
    out = tmp_path / "ket_qua.mp4"
    result = count_video(get_detector(), synthetic_video, output_path=out, label_fn=vi_label)
    assert result.output_path == out and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode().lower()
    assert fourcc in {"avc1", "h264"}  # H.264: trình duyệt phát được


def test_vid_stride_processes_fewer_frames(synthetic_video):
    assert count_video(get_detector(), synthetic_video, vid_stride=2).frames_processed == 15


def test_short_tracks_are_ignored(synthetic_video):
    assert count_video(get_detector(), synthetic_video, min_track_frames=1000).total == 0


def test_progress_reports_every_frame(synthetic_video):
    calls = []
    count_video(get_detector(), synthetic_video, progress=lambda done, total: calls.append((done, total)))
    assert len(calls) == 30 and calls[-1] == (30, 30)


def test_invalid_video_raises(tmp_path):
    bad = tmp_path / "hong.mp4"
    bad.write_text("khong phai video")
    with pytest.raises(InvalidVideoError, match="không phải là file video"):
        count_video(get_detector(), bad)


def test_invalid_settings_raise(synthetic_video):
    with pytest.raises(ValueError):
        count_video(get_detector(), synthetic_video, confidence=2)
    with pytest.raises(ValueError):
        count_video(get_detector(), synthetic_video, vid_stride=0)


def test_image_detection_unchanged_after_video(synthetic_video):
    detector = get_detector()
    before = detector.detect(ASSETS / "bus.jpg").counts
    count_video(detector, synthetic_video)
    assert detector.detect(ASSETS / "bus.jpg").counts == before


def test_write_video_csv(tmp_path):
    result = VideoCountResult(counts={"person": 3, "car": 1}, frames_processed=30, fps=15, duration_s=2)
    path = write_video_csv(result, tmp_path / "video.csv")
    with open(path, encoding="utf-8-sig", newline="") as f:
        assert list(csv.reader(f)) == [
            ["Loại vật thể", "Tên gốc (model)", "Số vật khác nhau"],
            ["người", "person", "3"],
            ["ô tô", "car", "1"],
            ["Tổng cộng", "", "4"],
        ]
```

- [ ] **Bước 2:** Chạy `.venv/bin/pytest tests/test_video.py -q`. Kỳ vọng: FAIL với `ImportError: cannot import name 'InvalidVideoError'`.

- [ ] **Bước 3: Viết code**

Thêm vào cuối `vision_count/config.py`:

```python
# Đếm video: một ID phải xuất hiện ít nhất 3 khung hình mới được tính (bỏ nhận diện chập chờn)
MIN_TRACK_FRAMES = 3
# Thuật toán tracking có sẵn trong Ultralytics (ByteTrack: nhanh, hợp chạy CPU)
TRACKER = "bytetrack.yaml"
```

Thêm vào `requirements.txt` (dưới dòng gradio):

```
# API (đã được cài sẵn theo Gradio, ghi ra cho rõ)
fastapi>=0.110
uvicorn>=0.29
python-multipart>=0.0.9
# Tracking trong video (ByteTrack cần thư viện này)
lap>=0.5.12
```

Trong `vision_count/drawing.py`, tách phần vẽ nhãn trong vòng lặp của `draw_detections` thành hàm dùng chung (đặt trước `draw_detections`):

```python
def draw_label(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color, font, pad: int) -> None:
    """Vẽ nhãn chữ trắng trên nền màu, đặt phía trên điểm (x, y); sát mép trên thì đặt xuống dưới."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    label_y = y - text_h - 2 * pad if y - text_h - 2 * pad >= 0 else y
    draw.rectangle((x, label_y, x + text_w + 2 * pad, label_y + text_h + 2 * pad), fill=color)
    draw.text((x + pad, label_y + pad - top), text, fill=(255, 255, 255), font=font)
```

Trong vòng lặp của `draw_detections`, thay các dòng từ `left, top, right, bottom = draw.textbbox(...)` đến `draw.text(...)` bằng:

```python
        draw_label(draw, x1, y1, text, color, font, pad=max(2, line_width))
```

Trong `vision_count/history.py`, cho `add()` nhận cả kết quả video: thêm `from typing import Protocol` và

```python
class CountResult(Protocol):
    """Bất kỳ kết quả nào có số đếm theo loại và tổng (ảnh: DetectionResult, video: VideoCountResult)."""

    counts: dict[str, int]

    @property
    def total(self) -> int: ...
```

rồi đổi kiểu tham số `result: DetectionResult` của `add` thành `result: CountResult`. Bỏ import `DetectionResult` nếu không còn dùng.

`vision_count/video.py`:

```python
"""Đếm vật thể trong video, có tracking (ByteTrack) để KHÔNG đếm trùng.

Trong video, cùng một người xuất hiện ở hàng chục khung hình. Tracking gán cho mỗi vật một ID
cố định qua các khung; đếm số ID khác nhau là ra số vật thật đã xuất hiện.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from vision_count import config
from vision_count.detector import ObjectDetector
from vision_count.drawing import PALETTE, draw_label, find_font
from vision_count.export import write_csv
from vision_count.labels_vi import vi_label


class InvalidVideoError(ValueError):
    """Lỗi khi file đầu vào không phải video hợp lệ."""


@dataclass
class VideoCountResult:
    counts: dict[str, int] = field(default_factory=dict)  # số vật KHÁC NHAU theo loại
    frames_processed: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    peak_in_frame: int = 0  # nhiều nhất bao nhiêu vật cùng lúc trong một khung
    output_path: Path | None = None  # video đã vẽ khung + ID (nếu có yêu cầu)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _probe(video_path: str | Path) -> tuple[float, int]:
    """Đọc fps và số khung hình; file không đọc được thì báo lỗi rõ ràng."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened() or not cap.read()[0]:
            raise InvalidVideoError(
                f"'{Path(video_path).name}' không phải là file video hợp lệ (hỗ trợ MP4, MOV, AVI...)."
            )
        return cap.get(cv2.CAP_PROP_FPS) or 0.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _open_writer(path: Path, fps: float, shape: tuple[int, ...]) -> cv2.VideoWriter:
    height, width = shape[:2]
    for codec in ("avc1", "mp4v"):  # avc1 (H.264) phát được trên trình duyệt; mp4v là dự phòng
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), max(fps, 1.0), (width, height))
        if writer.isOpened():
            return writer
    raise RuntimeError("Không ghi được video kết quả trên máy này.")


def _draw_frame(bgr, boxes, ids, classes, names, label_fn, counted: int):
    """Vẽ khung + 'tên #ID' cho từng vật đang được theo dõi, và số đã đếm ở góc trên trái."""
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    scale = max(img.size) / 1000
    width = max(2, round(3 * scale))
    font = find_font(max(12, round(18 * scale)))
    for box, track_id, cls in zip(boxes, ids, classes):
        color = PALETTE[cls % len(PALETTE)]
        x1, y1, x2, y2 = (float(v) for v in box)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        name = label_fn(names[cls]) if label_fn else names[cls]
        draw_label(draw, x1, y1, f"{name} #{track_id}", color, font, pad=width)
    draw_label(draw, 0, 0, f"Đã đếm: {counted}", (0, 0, 0), font, pad=width * 2)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def count_video(
    detector: ObjectDetector,
    video_path: str | Path,
    confidence: float = config.DEFAULT_CONFIDENCE,
    classes: list[str] | None = None,
    vid_stride: int = 1,
    output_path: str | Path | None = None,
    label_fn: Callable[[str], str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    min_track_frames: int = config.MIN_TRACK_FRAMES,
) -> VideoCountResult:
    """Đếm số vật khác nhau trong video.

    vid_stride: chỉ xử lý 1 trên N khung hình (nhanh hơn với video dài, tracking kém chính xác hơn).
    output_path: nếu có, ghi video đã vẽ khung + ID ra file mp4.
    progress(done, total): được gọi sau mỗi khung đã xử lý (để hiện thanh tiến độ).
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Ngưỡng độ tin cậy phải nằm trong khoảng 0 đến 1.")
    if vid_stride < 1:
        raise ValueError("vid_stride phải từ 1 trở lên.")
    fps, total_frames = _probe(video_path)
    class_ids = detector._class_names_to_ids(classes) if classes else None

    from ultralytics import YOLO

    # Bản model riêng cho lần đếm này: trạng thái tracking không dính vào model dùng chung,
    # và an toàn khi đếm ảnh chạy song song ở luồng khác
    model = YOLO(str(detector.model_path))
    expected = max(1, math.ceil(total_frames / vid_stride))
    seen: dict[int, Counter] = defaultdict(Counter)  # ID -> số khung xuất hiện theo từng loại
    writer = None
    processed = peak = 0
    try:
        for r in model.track(
            source=str(video_path),
            stream=True,  # xử lý từng khung, không giữ cả video trong bộ nhớ
            persist=False,  # mỗi video bắt đầu tracking lại từ đầu
            conf=confidence,
            classes=class_ids,
            imgsz=config.DEFAULT_IMAGE_SIZE,
            tracker=config.TRACKER,
            vid_stride=vid_stride,
            device=detector.device,
            verbose=False,
        ):
            processed += 1
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else []
            boxes = r.boxes.xyxy.cpu().numpy() if ids else []
            clss = r.boxes.cls.int().tolist() if ids else []
            for track_id, cls in zip(ids, clss):
                seen[track_id][model.names[cls]] += 1
            peak = max(peak, len(ids))
            if output_path is not None:
                if writer is None:
                    writer = _open_writer(Path(output_path), fps / vid_stride, r.orig_img.shape)
                counted = sum(1 for c in seen.values() if sum(c.values()) >= min_track_frames)
                writer.write(_draw_frame(r.orig_img, boxes, ids, clss, model.names, label_fn, counted))
            if progress:
                progress(processed, expected)
    finally:
        if writer is not None:
            writer.release()

    counts: Counter[str] = Counter()
    for per_class in seen.values():
        if sum(per_class.values()) >= min_track_frames:  # bỏ ID chập chờn
            counts[per_class.most_common(1)[0][0]] += 1  # loại xuất hiện nhiều nhất của ID đó
    return VideoCountResult(
        counts=dict(counts.most_common()),
        frames_processed=processed,
        fps=fps,
        duration_s=round(total_frames / fps, 2) if fps else 0.0,
        peak_in_frame=peak,
        output_path=Path(output_path) if writer is not None else None,
    )


def write_video_csv(result: VideoCountResult, path: str | Path) -> Path:
    """Bảng số vật khác nhau theo loại, có dòng tổng cộng (utf-8-sig cho Excel)."""
    path = Path(path)
    rows = [[vi_label(label), label, n] for label, n in result.counts.items()]
    rows.append(["Tổng cộng", "", result.total])
    write_csv(path, ["Loại vật thể", "Tên gốc (model)", "Số vật khác nhau"], rows)
    return path
```

Trong `vision_count/__init__.py`, xuất thêm `InvalidVideoError`, `VideoCountResult`, `count_video`, `write_video_csv` (từ video) và `draw_label` (từ drawing).

- [ ] **Bước 4:** Chạy `.venv/bin/pytest -q`. Kỳ vọng: tất cả PASS.
- [ ] **Bước 5:** Commit `Thêm đếm video có tracking (ByteTrack), không đếm trùng`.

---

### Task 2: Giao diện: tab Video và ô lọc đổi theo model

**Files:** Sửa `ui/handlers.py`, `ui/gradio_app.py`, `tests/test_handlers.py`, `tests/test_ui.py`, `README.md`.

**Interfaces:**
- Consumes: `count_video`, `write_video_csv`, `InvalidVideoError` (Task 1).
- Produces:
  - `handlers.class_choices(model_key) -> dict`, là một `gr.update(choices=[(display_label(n), n), ...], value=[])`.
  - `handlers.count_video_file(video_path, model_key, confidence, selected_classes, vid_stride, history=None, progress=None) -> (summary, table_df, video_path_str, csv_path_str)`.
  - `handlers.VIDEO_COLUMNS = ["Loại vật thể", "Số vật khác nhau"]`.

- [ ] **Bước 1: Thêm test vào `tests/test_handlers.py`**

```python
# ---------- Video ----------

def test_count_video_file(history, synthetic_video):
    summary, table, video_out, csv_path = handlers.count_video_file(
        str(synthetic_video), "nano", 0.25, [], 1, history
    )
    assert "3 vật thể khác nhau" in summary
    assert table.values.tolist() == [["người (person)", 3]]
    assert Path(video_out).is_file() and Path(csv_path).is_file()
    [entry] = history.recent()
    assert entry.source == "synthetic.mp4" and entry.total == 3 and entry.mode.startswith("Video")


def test_count_video_file_without_video_raises():
    with pytest.raises(gr.Error):
        handlers.count_video_file(None, "nano", 0.25, [], 1, None)


def test_count_video_file_bad_video_raises(tmp_path):
    bad = tmp_path / "hong.mp4"
    bad.write_text("x")
    with pytest.raises(gr.Error):
        handlers.count_video_file(str(bad), "nano", 0.25, [], 1, None)


# ---------- Ô lọc đổi theo model ----------

def test_class_choices_follow_model():
    update = handlers.class_choices("small")
    assert update["choices"][0] == ("người (person)", "person")
    assert len(update["choices"]) == 80
    assert update["value"] == []
```

Trong `tests/test_ui.py`, sửa tập tab mong đợi thành `{"Một ảnh", "Nhiều ảnh", "Video", "Lịch sử"}`.

- [ ] **Bước 2:** Chạy `.venv/bin/pytest tests/test_handlers.py tests/test_ui.py -q`. Kỳ vọng: FAIL (`AttributeError: module 'ui.handlers' has no attribute 'count_video_file'`; thiếu tab Video).

- [ ] **Bước 3: Viết code**

`ui/handlers.py`: import thêm `InvalidVideoError`, `count_video`, `write_video_csv` từ `vision_count`. Thêm hằng số `VIDEO_COLUMNS = ["Loại vật thể", "Số vật khác nhau"]`. Thêm 2 hàm:

```python
# ---------- Ô lọc theo model ----------

def class_choices(model_key):
    """Đổi model thì cập nhật ô 'Chỉ đếm các loại' theo các loại model đó nhận được (vd model fine-tune)."""
    detector = _load_model(model_key)
    return gr.update(choices=[(display_label(n), n) for n in detector.class_names], value=[])


# ---------- Video ----------

def count_video_file(
    video_path, model_key, confidence, selected_classes, vid_stride, history: HistoryStore | None = None, progress=None
):
    """Nút 'Đếm video'. Trả về (tóm tắt, bảng, video kết quả, file CSV)."""
    if not video_path:
        raise gr.Error("Vui lòng tải một video lên trước.")
    detector = _load_model(model_key)
    stride = int(vid_stride)
    out_dir = new_export_dir()

    def report(done, total):
        if progress is not None:
            progress(done / total, desc=f"Đang xử lý khung hình {done}/{total}")

    try:
        result = count_video(
            detector,
            video_path,
            confidence=confidence,
            classes=selected_classes or None,
            vid_stride=stride,
            output_path=out_dir / "video_ket_qua.mp4",
            label_fn=vi_label,
            progress=report,
        )
    except (InvalidVideoError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc

    mode = "Video" + (f", 1/{stride} khung hình" if stride > 1 else "")
    headline = (
        f"### {result.total} vật thể khác nhau trong video"
        if result.total
        else "### Không đếm được vật thể nào trong video\nThử giảm ngưỡng độ tin cậy hoặc bỏ bộ lọc loại vật."
    )
    summary = (
        f"{headline}\n"
        f"{result.frames_processed} khung hình đã xử lý · video dài {result.duration_s:.1f} giây · "
        f"nhiều nhất {result.peak_in_frame} vật cùng lúc\n"
        f"_Model: {_model_name(model_key)} · Chế độ: {mode}_"
    )
    table = pd.DataFrame([[display_label(l), n] for l, n in result.counts.items()], columns=VIDEO_COLUMNS)
    csv_path = write_video_csv(result, out_dir / "so_luong_video.csv")
    if history is not None:
        history.add(Path(video_path).name, model_key, mode, result)
    video_out = str(result.output_path) if result.output_path else None
    return summary, table, video_out, str(csv_path)
```

`ui/gradio_app.py`:
- Import thêm `VIDEO_COLUMNS`.
- Thêm tab Video giữa "Nhiều ảnh" và "Lịch sử":

```python
            # ---------- Tab Video ----------
            with gr.Tab("Video"):
                gr.Markdown(
                    "Đếm số vật **khác nhau** xuất hiện trong video: mỗi vật được gán một ID và theo dõi qua "
                    "các khung hình, nên không bị đếm trùng. Dùng Model, Ngưỡng và Chỉ đếm các loại ở phần "
                    "Cài đặt (chế độ vật nhỏ không áp dụng cho video)."
                )
                # Chỉ cho tải file lên: không cần ffmpeg (quay webcam thì Gradio cần ffmpeg để xử lý)
                video_input = gr.Video(label="Video cần đếm", sources=["upload"])
                stride_input = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=1,
                    label="Xử lý 1 trên N khung hình",
                    info="Tăng lên để chạy nhanh hơn với video dài (tracking có thể kém chính xác hơn).",
                )
                video_button = gr.Button("Đếm video", variant="primary")
                video_summary = gr.Markdown()
                video_table = gr.Dataframe(headers=VIDEO_COLUMNS, label="Số vật khác nhau theo loại", interactive=False)
                video_output = gr.Video(label="Video kết quả (khung + ID)", interactive=False)
                video_download = gr.File(label="Tải bảng số lượng (CSV)", interactive=False)
```

- Nối sự kiện:

```python
        def on_video(path, model_key, conf, classes, stride, progress=gr.Progress()):
            return handlers.count_video_file(path, model_key, conf, classes, stride, history, progress)

        video_button.click(
            fn=on_video,
            inputs=[video_input, model_input, confidence_input, class_input, stride_input],
            outputs=[video_summary, video_table, video_output, video_download],
        )
        # Đổi model thì ô lọc đổi theo các loại của model đó (quan trọng với model fine-tune)
        model_input.change(fn=handlers.class_choices, inputs=model_input, outputs=class_input)
```

README:
- Thêm mục "### Đếm video" vào phần Cách dùng: tải video lên, bấm Đếm video, xem số vật khác nhau, video kết quả có khung + "tên #ID" và dòng "Đã đếm: N" ở góc, tải CSV.
  - Giải thích tracking: mỗi vật một ID; một ID phải xuất hiện ít nhất 3 khung mới tính.
  - Thanh trượt "Xử lý 1 trên N khung hình"; tốc độ tham khảo khoảng 0.05 giây/khung trên Mac M1 với Nano, tức video 1 phút ở 30 fps mất khoảng 1.5-2 phút.
  - Giới hạn: người ra khỏi khung hình rồi quay lại có thể bị tính là người mới; vật bị che lâu có thể bị cấp ID mới.
- Cây thư mục: thêm `video.py`.
- Mục 5 (code): ví dụ `count_video(get_detector(), "video.mp4", output_path="ket_qua.mp4")`.
- Nói rõ ô lọc tự đổi theo model.

- [ ] **Bước 4:** Chạy `.venv/bin/pytest -q`. Kỳ vọng: tất cả PASS.
- [ ] **Bước 5: Kiểm tra trên trình duyệt:** tab Video, tải video tự tạo, bấm Đếm video → "3 vật thể khác nhau", video kết quả phát được. Đổi model sang Small → ô lọc vẫn có 80 loại, các lựa chọn đã chọn được xóa.
- [ ] **Bước 6:** Commit `Giao diện: tab Video đếm có tracking; ô lọc đổi theo model`.

---

### Task 3: API FastAPI (`api/main.py`)

**Files:** Tạo `api/__init__.py` (rỗng), `api/main.py`, `tests/test_api.py`. Sửa `README.md`.

**Interfaces:**
- Consumes: `get_detector`, `load_image`, `filter_by_region`, `draw_detections`, `count_video`, `vi_label`, `config.AVAILABLE_MODELS` và các lỗi (`InvalidImageError`, `InvalidVideoError`).
- Produces (HTTP):
  - `GET /health` → `{"status": "ok"}`.
  - `GET /models` → `[{"key", "name", "downloaded"}]`.
  - `GET /classes?model=nano` → `[{"label", "label_vi"}]`.
  - `POST /detect` (multipart: `file`; form: `model`, `confidence`, `classes` dạng "person,car", `tiled`, `region` dạng "x1,y1,x2,y2") → JSON gồm `model, total, counts, counts_vi, detections[{label, label_vi, class_id, confidence, box}], image_width, image_height, region`.
  - `POST /detect/image` (như trên + `label_style`) → `image/jpeg`.
  - `POST /video/count` (multipart: `file`; form: `model`, `confidence`, `classes`, `vid_stride`) → JSON gồm `model, total, counts, counts_vi, frames_processed, fps, duration_s, peak_in_frame`.
  - Lỗi đầu vào → 400 `{"detail": "<tiếng Việt>"}`; ngưỡng ngoài [0, 1] → 422.

- [ ] **Bước 1: Viết test `tests/test_api.py`**

```python
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


def test_api_video_rejects_non_video(tmp_path):
    bad = tmp_path / "hong.mp4"
    bad.write_text("x")
    with open(bad, "rb") as f:
        resp = client.post("/video/count", files={"file": ("hong.mp4", f, "video/mp4")})
    assert resp.status_code == 400 and "không phải là file video" in resp.json()["detail"]
```

- [ ] **Bước 2:** Chạy `.venv/bin/pytest tests/test_api.py -q`. Kỳ vọng: FAIL với `ModuleNotFoundError: No module named 'api'`.

- [ ] **Bước 3: Viết `api/main.py`**

```python
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
        image = load_image(file.file.read())
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
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / (Path(file.filename or "video.mp4").name)
        with open(path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        try:
            # count_video dùng bản model riêng nên không cần khóa
            result = count_video(
                get_detector(model), path, confidence=confidence, classes=_parse_classes(classes), vid_stride=vid_stride
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
    }
```

Thêm mục "## 6. API cho ứng dụng khác (FastAPI)" vào README (trước "Lỗi thường gặp", đánh lại số các mục sau), gồm:
- Lệnh chạy `uvicorn api.main:app --port 8000`.
- Trang `/docs` để thử API ngay trên trình duyệt.
- Bảng các endpoint.
- Ví dụ `curl`:

```bash
curl -F "file=@samples/bus.jpg" http://127.0.0.1:8000/detect
curl -F "file=@samples/bus.jpg" -F "classes=person" -F "region=0,0,400,1080" http://127.0.0.1:8000/detect
curl -F "file=@samples/bus.jpg" -F "label_style=number" http://127.0.0.1:8000/detect/image -o ket_qua.jpg
curl -F "file=@video.mp4" http://127.0.0.1:8000/video/count
```

Ghi chú thêm: API chỉ mở cho máy mình (127.0.0.1). Muốn điện thoại cùng Wi-Fi gọi được thì chạy với `--host 0.0.0.0`, nhưng khi đó ai cùng mạng cũng gọi được.

- [ ] **Bước 4:** Chạy `.venv/bin/pytest -q`. Kỳ vọng: tất cả PASS.
- [ ] **Bước 5: Kiểm tra server thật:** chạy `.venv/bin/uvicorn api.main:app --port 8000`, `curl /health`, `curl -F file=@samples/bus.jpg /detect`, mở `/docs`. Kỳ vọng JSON có `"total": 5` và trang docs mở được.
- [ ] **Bước 6:** Commit `Thêm API FastAPI: detect, detect/image, video/count, models, classes`.

---

### Task 4: Bộ công cụ fine-tune (`training/`)

**Files:** Tạo `training/README.md`, `training/data.yaml`, `training/check_dataset.py`, `training/train_colab.ipynb`, `tests/test_check_dataset.py`. Sửa `README.md` mục fine-tune.

**Interfaces:**
- Produces:
  - `check_dataset(root: str | Path) -> tuple[list[str], list[str]]` trả về (lỗi, cảnh báo).
  - Dòng lệnh: `python training/check_dataset.py <thư_mục>`, mã thoát 0 nếu không có lỗi, 1 nếu có lỗi.

Cấu trúc bộ dữ liệu chuẩn (ghi trong `training/README.md`):

```
dataset/
├── data.yaml
├── images/
│   ├── train/   anh001.jpg ...
│   └── val/
└── labels/
    ├── train/   anh001.txt ...  (mỗi dòng: class_id x_center y_center width height, giá trị 0..1)
    └── val/
```

- [ ] **Bước 1: Viết test `tests/test_check_dataset.py`**

```python
"""Test script kiểm tra bộ dữ liệu fine-tune."""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from training.check_dataset import check_dataset

ROOT = Path(__file__).resolve().parent.parent


def _make_dataset(root: Path, labels: dict[str, str], names=("oc_vit", "dai_oc")):
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8",
    )
    for rel, text in labels.items():  # rel: "train/a" -> images/train/a.jpg + labels/train/a.txt
        split, stem = rel.split("/")
        Image.new("RGB", (64, 64)).save(root / "images" / split / f"{stem}.jpg")
        if text is not None:
            (root / "labels" / split / f"{stem}.txt").write_text(text)
    return root


def test_valid_dataset_has_no_errors(tmp_path):
    root = _make_dataset(tmp_path, {"train/a": "0 0.5 0.5 0.2 0.3\n1 0.1 0.1 0.05 0.05\n", "val/b": "1 0.5 0.5 0.4 0.4\n"})
    errors, warnings = check_dataset(root)
    assert errors == [] and warnings == []


def test_reports_bad_label_lines(tmp_path):
    root = _make_dataset(tmp_path, {
        "train/a": "5 0.5 0.5 0.2 0.3\n",  # class 5 không có trong names (chỉ có 0, 1)
        "train/b": "0 1.5 0.5 0.2 0.3\n",  # tọa độ ngoài 0..1
        "train/c": "0 0.5 0.5\n",  # thiếu số
        "val/d": "0 0.5 0.5 0.2 0.3\n",
    })
    errors, _ = check_dataset(root)
    joined = "\n".join(errors)
    assert "a.txt" in joined and "class" in joined
    assert "b.txt" in joined and "0..1" in joined
    assert "c.txt" in joined and "5 số" in joined


def test_warns_image_without_label_and_errors_label_without_image(tmp_path):
    root = _make_dataset(tmp_path, {"train/a": None, "val/b": "0 0.5 0.5 0.2 0.2\n"})
    (root / "labels" / "train" / "mo_coi.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    errors, warnings = check_dataset(root)
    assert any("a.jpg" in w for w in warnings)  # ảnh không có nhãn = ảnh nền (cho phép, nhưng nhắc)
    assert any("mo_coi.txt" in e for e in errors)


def test_missing_data_yaml_or_empty_val(tmp_path):
    errors, _ = check_dataset(tmp_path)
    assert any("data.yaml" in e for e in errors)
    root = _make_dataset(tmp_path / "ds", {"train/a": "0 0.5 0.5 0.2 0.2\n"})
    errors, _ = check_dataset(root)
    assert any("val" in e for e in errors)


def test_command_line_exit_code(tmp_path):
    root = _make_dataset(tmp_path, {"train/a": "0 0.5 0.5 0.2 0.2\n", "val/b": "0 0.5 0.5 0.2 0.2\n"})
    ok = subprocess.run([sys.executable, ROOT / "training" / "check_dataset.py", str(root)], capture_output=True, text=True)
    assert ok.returncode == 0 and "Không có lỗi" in ok.stdout
    bad = subprocess.run([sys.executable, ROOT / "training" / "check_dataset.py", str(tmp_path / "khong_co")], capture_output=True, text=True)
    assert bad.returncode == 1


def test_colab_notebook_is_valid():
    nb = json.loads((ROOT / "training" / "train_colab.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "model.train(" in code and "best.pt" in code


def test_data_yaml_template_parses():
    import yaml

    data = yaml.safe_load((ROOT / "training" / "data.yaml").read_text(encoding="utf-8"))
    assert {"train", "val", "names"} <= set(data)
```

Thêm `training/__init__.py` rỗng để test import được.

- [ ] **Bước 2:** Chạy `.venv/bin/pytest tests/test_check_dataset.py -q`. Kỳ vọng: FAIL với `ModuleNotFoundError: No module named 'training'`.

- [ ] **Bước 3: Viết code**

`training/check_dataset.py`:

```python
"""Kiểm tra bộ dữ liệu YOLO trước khi train, để không mất công train rồi mới biết dữ liệu sai.

Chạy: python training/check_dataset.py duong_dan/dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _class_count(names) -> int:
    return len(names) if isinstance(names, (list, dict)) else 0


def _check_label_file(path: Path, n_classes: int) -> list[str]:
    errors = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        where = f"{path.name} dòng {i}"
        if len(parts) != 5:
            errors.append(f"{where}: cần đúng 5 số (class x y w h), đang có {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{where}: có giá trị không phải số")
            continue
        if not 0 <= cls < n_classes:
            errors.append(f"{where}: class {cls} không có trong names của data.yaml (0..{n_classes - 1})")
        if not all(0 <= v <= 1 for v in (x, y, w, h)):
            errors.append(f"{where}: tọa độ phải nằm trong 0..1 (đã chia cho chiều rộng/cao ảnh)")
        elif w <= 0 or h <= 0:
            errors.append(f"{where}: chiều rộng/cao của khung phải lớn hơn 0")
    return errors


def check_dataset(root: str | Path) -> tuple[list[str], list[str]]:
    """Trả về (danh sách lỗi, danh sách cảnh báo). Không có lỗi thì có thể train."""
    root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        return [f"Không tìm thấy {yaml_path} (file khai báo đường dẫn và tên các loại)"], warnings
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    n_classes = _class_count(data.get("names"))
    if n_classes == 0:
        errors.append("data.yaml thiếu mục 'names' (danh sách tên các loại vật)")

    for split in ("train", "val"):
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        images = {p.stem: p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS} if image_dir.is_dir() else {}
        labels = {p.stem: p for p in label_dir.glob("*.txt")} if label_dir.is_dir() else {}
        if not images:
            errors.append(f"Thư mục images/{split} chưa có ảnh nào")
        for stem, image in sorted(images.items()):
            if stem not in labels:
                warnings.append(f"{split}: ảnh {image.name} không có file nhãn (sẽ được coi là ảnh nền, không có vật)")
        for stem, label in sorted(labels.items()):
            if stem not in images:
                errors.append(f"{split}: file nhãn {label.name} không có ảnh tương ứng")
            errors.extend(f"{split}: {e}" for e in _check_label_file(label, n_classes))
    return errors, warnings


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Cách dùng: python training/check_dataset.py <thư_mục_dataset>")
        return 1
    errors, warnings = check_dataset(argv[1])
    for w in warnings:
        print(f"⚠️  {w}")
    for e in errors:
        print(f"❌ {e}")
    if errors:
        print(f"\nCó {len(errors)} lỗi: sửa xong rồi chạy lại.")
        return 1
    print(f"\n✅ Không có lỗi ({len(warnings)} cảnh báo). Bộ dữ liệu sẵn sàng để train.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

`training/data.yaml`:

```yaml
# Mẫu file khai báo bộ dữ liệu YOLO. Đặt file này trong thư mục dataset/.
# Đường dẫn train/val tính từ "path" (dấu chấm = chính thư mục chứa file này).
path: .
train: images/train
val: images/val

# Tên các loại vật, đánh số từ 0. Thứ tự PHẢI khớp với số class trong các file nhãn .txt.
# Đặt tên tiếng Việt không dấu hoặc có dấu đều được; tên này sẽ hiện trên giao diện.
names:
  0: oc_vit
  1: dai_oc
```

`training/train_colab.ipynb`: notebook nbformat 4, gồm các ô sau theo thứ tự:
1. Markdown: giới thiệu và các bước.
2. Code `!nvidia-smi` (kiểm tra đã bật GPU: Runtime → Change runtime type → T4 GPU).
3. Code `!pip install -q ultralytics`.
4. Markdown: chuẩn bị file `dataset.zip` (nén thư mục `dataset/` có `data.yaml`).
5. Code: tải zip lên bằng `from google.colab import files; uploaded = files.upload()` rồi `!unzip -q -o dataset.zip -d /content/`.
6. Code: kiểm tra nhanh có `data.yaml`, đếm số ảnh train/val.
7. Code train: `from ultralytics import YOLO; model = YOLO("yolo11n.pt"); model.train(data="/content/dataset/data.yaml", epochs=100, imgsz=640, patience=20)`.
8. Code đánh giá: `metrics = model.val(); print(metrics.box.map50, metrics.box.map)`, kèm giải thích mAP50 bằng markdown.
9. Code thử dự đoán một ảnh val và hiển thị.
10. Code tải về: `files.download(".../weights/best.pt")`.
11. Markdown: đặt `best.pt` vào `models/`, thêm dòng vào `AVAILABLE_MODELS`.

Tạo file notebook bằng một đoạn Python dùng `json.dump` (để chắc chắn JSON hợp lệ), mỗi ô có `cell_type`, `metadata`, `source` (và với ô code thêm `execution_count: None`, `outputs: []`); `metadata` của notebook có `"accelerator": "GPU"` và `kernelspec` python3.

`training/README.md`: hướng dẫn chi tiết bằng tiếng Việt:
1. **Thu thập ảnh:** bao nhiêu ảnh, chụp đa dạng, có ảnh nền không có vật.
2. **Gán nhãn:** Label Studio chạy trên máy (`pip install label-studio` trong một môi trường ảo **riêng** để không xung đột thư viện; `label-studio start`), CVAT hoặc Roboflow; xuất định dạng **YOLO**.
3. **Sắp xếp thư mục** theo cấu trúc chuẩn ở trên; chia khoảng 80% train / 20% val.
4. **Kiểm tra:** `python training/check_dataset.py dataset`.
5. **Train trên Colab:** mở `train_colab.ipynb` (File → Upload notebook), chạy lần lượt các ô; khoảng 100 epoch với vài trăm ảnh trên GPU T4 mất khoảng 15-40 phút.
6. **Đọc kết quả:** mAP50 trên 0.8 là tốt; dưới 0.5 thì cần thêm ảnh hoặc sửa nhãn.
7. **Dùng trong vision-count:** chép `best.pt` vào `models/`, thêm `"cua_toi": ("Model của tôi", "best.pt"),` vào `AVAILABLE_MODELS`, chạy lại `python app.py`, chọn model trong ô Model, ô lọc tự đổi theo các loại mới. Tên tiếng Việt: đặt ngay trong `names` của `data.yaml`.

README chính, mục fine-tune: rút gọn, trỏ tới `training/README.md`. Cây thư mục thêm `training/` và `api/`.

- [ ] **Bước 4:** Chạy `.venv/bin/pytest -q`. Kỳ vọng: tất cả PASS.
- [ ] **Bước 5:** Commit `Thêm bộ công cụ fine-tune: hướng dẫn, data.yaml mẫu, notebook Colab, script kiểm tra dữ liệu`.
