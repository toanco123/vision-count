"""Đếm vật thể trong video, có tracking (ByteTrack) để KHÔNG đếm trùng.

Trong video, cùng một người xuất hiện ở hàng chục khung hình. Tracking gán cho mỗi vật một ID
cố định qua các khung; đếm số ID khác nhau là ra số vật thật đã xuất hiện.
"""

from __future__ import annotations

import math
import tempfile
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


class VideoWriteError(ValueError):
    """Lỗi khi không ghi được video kết quả."""


@dataclass
class VideoCountResult:
    counts: dict[str, int] = field(default_factory=dict)  # số vật KHÁC NHAU theo loại
    frames_processed: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    peak_in_frame: int = 0  # nhiều nhất bao nhiêu vật cùng lúc trong một khung
    output_path: Path | None = None  # video đã vẽ khung + ID (nếu có yêu cầu)
    frames_expected: int = 0  # số khung lẽ ra phải xử lý (0 = video không cho biết số khung)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def complete(self) -> bool:
        """Đã đọc được gần hết video chưa (video hỏng giữa chừng sẽ dừng đọc sớm)."""
        return self.frames_expected == 0 or self.frames_processed >= 0.9 * self.frames_expected

    @property
    def warning(self) -> str | None:
        if self.complete:
            return None
        return (
            f"Chỉ đọc được {self.frames_processed}/{self.frames_expected} khung hình: video có thể bị hỏng, "
            "kết quả chỉ tính phần đã đọc được."
        )


# Giá trị mặc định lấy từ cfg/trackers/bytetrack.yaml của Ultralytics
_BYTETRACK_DEFAULTS = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}


def _tracker_config(confidence: float) -> str:
    """File cấu hình ByteTrack. Mặc định ByteTrack chỉ tạo ID mới cho khung có độ tin cậy >= 0.25,
    nên khi người dùng hạ ngưỡng dưới 0.25 thì hạ theo, để các vật mờ cũng được theo dõi."""
    cfg = dict(_BYTETRACK_DEFAULTS)
    if confidence < cfg["new_track_thresh"]:
        cfg["track_high_thresh"] = cfg["new_track_thresh"] = confidence
        cfg["track_low_thresh"] = min(cfg["track_low_thresh"], confidence)
    path = Path(tempfile.gettempdir()) / f"vision_count_bytetrack_{confidence:.3f}.yaml"
    path.write_text("".join(f"{k}: {v}\n" for k, v in cfg.items()), encoding="utf-8")
    return str(path)


def _effective_min_frames(min_track_frames: int, vid_stride: int) -> int:
    """Số khung ĐÃ XỬ LÝ tối thiểu. Bỏ bớt khung (stride) thì cần ít khung hơn cho cùng thời lượng,
    nhưng vẫn ít nhất 2 (nếu yêu cầu từ 2 trở lên) để bỏ ID chỉ thấy 1 khung."""
    return max(min(2, min_track_frames), math.ceil(min_track_frames / vid_stride))


def _check_format(video_path: str | Path, name: str) -> None:
    """Chỉ nhận đuôi file mà Ultralytics đọc được; đuôi khác Ultralytics sẽ báo lỗi khó hiểu."""
    from ultralytics.data.utils import VID_FORMATS

    suffix = Path(video_path).suffix.lower().lstrip(".")
    if suffix not in VID_FORMATS:
        kind = f"đuôi '.{suffix}'" if suffix else "không có đuôi file"
        raise InvalidVideoError(
            f"'{name}' {kind} chưa được hỗ trợ. Hãy dùng video có đuôi: {', '.join(sorted(VID_FORMATS))}."
        )


def _is_image(video_path: str | Path) -> bool:
    try:
        with Image.open(video_path) as img:
            img.verify()
        return True
    except Exception:  # Pillow không mở được: không phải ảnh
        return False


def _probe(video_path: str | Path, name: str | None = None) -> tuple[float, int]:
    """Đọc fps và số khung hình; file không đọc được thì báo lỗi rõ ràng."""
    name = name or Path(video_path).name
    # Ảnh bị đổi đuôi thành .mp4 vẫn mở được bằng OpenCV, nên kiểm tra riêng (trừ gif, vốn là video)
    if Path(video_path).suffix.lower() != ".gif" and _is_image(video_path):
        raise InvalidVideoError(f"'{name}' là file ảnh, không phải video. Hãy đếm ảnh ở tab Một ảnh (hoặc API /detect).")
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened() or not cap.read()[0]:
            raise InvalidVideoError(f"'{name}' không phải là file video hợp lệ (hỗ trợ MP4, MOV, AVI...).")
        # Một số video không ghi số khung (trả về 0 hoặc số âm): coi như "không biết"
        return cap.get(cv2.CAP_PROP_FPS) or 0.0, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()


def _open_writer(path: Path, fps: float, shape: tuple[int, ...]) -> cv2.VideoWriter:
    height, width = shape[:2]
    for codec in ("avc1", "mp4v"):  # avc1 (H.264) phát được trên trình duyệt; mp4v là dự phòng
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), max(fps, 1.0), (width, height))
        if writer.isOpened():
            return writer
    raise VideoWriteError("Không ghi được video kết quả trên máy này.")


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
    name: str | None = None,
) -> VideoCountResult:
    """Đếm số vật khác nhau trong video.

    vid_stride: chỉ xử lý 1 trên N khung hình (nhanh hơn với video dài, tracking kém chính xác hơn).
    output_path: nếu có, ghi video đã vẽ khung + ID ra file mp4.
    progress(done, total): được gọi sau mỗi khung đã xử lý (để hiện thanh tiến độ).
    name: tên hiển thị trong thông báo lỗi (vd tên file người dùng tải lên qua API).
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Ngưỡng độ tin cậy phải nằm trong khoảng 0 đến 1.")
    if vid_stride < 1:
        raise ValueError("vid_stride phải từ 1 trở lên.")
    name = name or Path(video_path).name
    _check_format(video_path, name)
    fps, total_frames = _probe(video_path, name)
    class_ids = detector._class_names_to_ids(classes) if classes else None

    from ultralytics import YOLO

    # Bản model riêng cho lần đếm này: trạng thái tracking không dính vào model dùng chung,
    # và an toàn khi đếm ảnh chạy song song ở luồng khác
    model = YOLO(str(detector.model_path))
    expected = total_frames // vid_stride  # Ultralytics bỏ qua khung theo cùng cách tính này
    seen: dict[int, Counter] = defaultdict(Counter)  # ID -> số khung xuất hiện theo từng loại
    frame_ids: list[list[int]] = []  # các ID có mặt ở từng khung (để tính "nhiều nhất cùng lúc")
    min_frames = _effective_min_frames(min_track_frames, vid_stride)
    writer = None
    processed = 0
    try:
        for r in model.track(
            source=str(video_path),
            stream=True,  # xử lý từng khung, không giữ cả video trong bộ nhớ
            persist=False,  # mỗi video bắt đầu tracking lại từ đầu
            conf=confidence,
            classes=class_ids,
            imgsz=config.DEFAULT_IMAGE_SIZE,
            tracker=_tracker_config(confidence),
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
            frame_ids.append(ids)
            if output_path is not None:
                if writer is None:
                    writer = _open_writer(Path(output_path), fps / vid_stride, r.orig_img.shape)
                counted = sum(1 for c in seen.values() if sum(c.values()) >= min_frames)
                writer.write(_draw_frame(r.orig_img, boxes, ids, clss, model.names, label_fn, counted))
            if progress:
                progress(processed, max(expected, processed))  # không biết số khung thì không vượt 100%
    finally:
        if writer is not None:
            writer.release()

    counted_ids = {tid for tid, per_class in seen.items() if sum(per_class.values()) >= min_frames}  # bỏ ID chập chờn
    counts: Counter[str] = Counter(seen[tid].most_common(1)[0][0] for tid in counted_ids)  # loại thấy nhiều nhất
    # Chỉ tính các vật đã được đếm, để "nhiều nhất cùng lúc" không lớn hơn tổng
    peak = max((len(counted_ids.intersection(ids)) for ids in frame_ids), default=0)
    return VideoCountResult(
        counts=dict(counts.most_common()),
        frames_processed=processed,
        fps=fps,
        duration_s=round((total_frames or processed * vid_stride) / fps, 2) if fps else 0.0,
        peak_in_frame=peak,
        output_path=Path(output_path) if writer is not None else None,
        frames_expected=expected,
    )


def write_video_csv(result: VideoCountResult, path: str | Path) -> Path:
    """Bảng số vật khác nhau theo loại, có dòng tổng cộng (utf-8-sig cho Excel)."""
    path = Path(path)
    rows = [[vi_label(label), label, n] for label, n in result.counts.items()]
    rows.append(["Tổng cộng", "", result.total])
    write_csv(path, ["Loại vật thể", "Tên gốc (model)", "Số vật khác nhau"], rows)
    return path
