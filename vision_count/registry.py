"""Nạp và giữ lại các model, để đổi model qua lại không phải nạp lại từ đầu."""

from __future__ import annotations

import threading
from functools import lru_cache

from vision_count import config
from vision_count.detector import ObjectDetector


# 2 yêu cầu cùng lúc ở lần đầu (vd Gradio và API): chỉ nạp/tải model một lần
_load_lock = threading.Lock()


def get_detector(model_key: str = config.DEFAULT_MODEL_KEY) -> ObjectDetector:
    """Trả về detector của model được chọn. Mỗi model chỉ nạp một lần (an toàn khi gọi từ nhiều luồng)."""
    if model_key not in config.AVAILABLE_MODELS:
        raise ValueError(f"Không có model '{model_key}'. Chọn một trong: {', '.join(config.AVAILABLE_MODELS)}")
    with _load_lock:
        return _load_detector(model_key)


# Cache tách riêng và chỉ nhận khóa dạng chuỗi: nếu đặt lru_cache thẳng lên get_detector thì
# get_detector() và get_detector("nano") bị coi là 2 khóa khác nhau, model bị nạp 2 lần.
@lru_cache(maxsize=None)
def _load_detector(model_key: str) -> ObjectDetector:
    _, weights = config.AVAILABLE_MODELS[model_key]
    return ObjectDetector(weights)


def is_model_downloaded(model_key: str) -> bool:
    """Model đã có sẵn trong thư mục models/ chưa (chưa có thì lần đầu dùng sẽ phải tải)."""
    if model_key not in config.AVAILABLE_MODELS:
        return False
    _, weights = config.AVAILABLE_MODELS[model_key]
    return (config.MODELS_DIR / weights).is_file()
