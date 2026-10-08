"""Nạp và giữ lại các model, để đổi model qua lại không phải nạp lại từ đầu."""

from __future__ import annotations

from functools import lru_cache

from vision_count import config
from vision_count.detector import ObjectDetector


@lru_cache(maxsize=None)
def get_detector(model_key: str = config.DEFAULT_MODEL_KEY) -> ObjectDetector:
    """Trả về detector của model được chọn. Mỗi model chỉ nạp một lần."""
    if model_key not in config.AVAILABLE_MODELS:
        raise ValueError(f"Không có model '{model_key}'. Chọn một trong: {', '.join(config.AVAILABLE_MODELS)}")
    _, weights = config.AVAILABLE_MODELS[model_key]
    return ObjectDetector(weights)


def is_model_downloaded(model_key: str) -> bool:
    """Model đã có sẵn trong thư mục models/ chưa (chưa có thì lần đầu dùng sẽ phải tải)."""
    if model_key not in config.AVAILABLE_MODELS:
        return False
    _, weights = config.AVAILABLE_MODELS[model_key]
    return (config.MODELS_DIR / weights).is_file()
