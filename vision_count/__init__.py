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
from vision_count.drawing import draw_detections
from vision_count.labels_vi import display_label, vi_label

__all__ = [
    "Detection",
    "DetectionResult",
    "InvalidImageError",
    "ObjectDetector",
    "display_label",
    "draw_detections",
    "load_image",
    "vi_label",
]
