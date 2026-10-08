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
