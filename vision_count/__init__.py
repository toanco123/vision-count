"""vision_count: phần xử lý AI (không phụ thuộc giao diện).

Gradio, FastAPI hay bất kỳ giao diện nào khác đều chỉ cần gọi:

    from vision_count import ObjectDetector
    detector = ObjectDetector()
    result = detector.detect("anh.jpg")
"""

from vision_count.detector import (
    Detection,
    DetectionResult,
    InvalidImageError,
    ObjectDetector,
    load_image,
)
from vision_count.batch import BatchItem, count_many, write_batch_csv
from vision_count.drawing import (
    LABEL_FULL,
    LABEL_NONE,
    LABEL_NUMBER,
    LABEL_STYLES,
    draw_detections,
    draw_label,
    draw_region,
)
from vision_count.export import export_result
from vision_count.history import HistoryEntry, HistoryStore
from vision_count.labels_vi import display_label, vi_label
from vision_count.region import Region, filter_by_region, normalize_region
from vision_count.registry import get_detector, is_model_downloaded
from vision_count.video import InvalidVideoError, VideoCountResult, count_video, write_video_csv

__all__ = [
    "LABEL_FULL",
    "LABEL_NONE",
    "LABEL_NUMBER",
    "LABEL_STYLES",
    "BatchItem",
    "Detection",
    "DetectionResult",
    "HistoryEntry",
    "HistoryStore",
    "InvalidImageError",
    "InvalidVideoError",
    "ObjectDetector",
    "Region",
    "VideoCountResult",
    "count_many",
    "count_video",
    "display_label",
    "draw_detections",
    "draw_label",
    "draw_region",
    "export_result",
    "filter_by_region",
    "get_detector",
    "is_model_downloaded",
    "load_image",
    "normalize_region",
    "vi_label",
    "write_batch_csv",
    "write_video_csv",
]
