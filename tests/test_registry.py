"""Test chọn model qua get_detector."""

import pytest

from vision_count import config, get_detector, is_model_downloaded


def test_get_detector_caches_instance():
    # Đổi model qua lại không được nạp lại từ đầu (rất chậm)
    assert get_detector("nano") is get_detector("nano")


def test_small_model_uses_its_own_weights():
    detector = get_detector("small")  # lần đầu sẽ tải yolo11s.pt (~19MB)
    assert detector.model_path.name == "yolo11s.pt"
    assert detector is not get_detector("nano")
    assert len(detector.class_names) == 80


def test_unknown_model_key_raises():
    with pytest.raises(ValueError):
        get_detector("khong_ton_tai")


def test_is_model_downloaded():
    get_detector("nano")
    assert is_model_downloaded("nano") is True
    assert is_model_downloaded("khong_ton_tai") is False


def test_default_key_is_in_available_models():
    assert config.DEFAULT_MODEL_KEY in config.AVAILABLE_MODELS


def test_default_key_shares_cache_with_explicit_key():
    # get_detector() và get_detector("nano") phải là cùng một model, không nạp 2 lần
    assert get_detector() is get_detector("nano") is get_detector(model_key="nano")


def test_get_detector_loads_once_under_concurrency(monkeypatch):
    import threading
    import time

    from vision_count import registry

    built = []

    class SlowDetector:
        def __init__(self, weights):
            time.sleep(0.2)  # giả lập đang tải model
            built.append(weights)

    registry._load_detector.cache_clear()
    monkeypatch.setattr(registry, "ObjectDetector", SlowDetector)
    try:
        threads = [threading.Thread(target=registry.get_detector, args=("nano",)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert built == ["yolo11n.pt"]
    finally:
        registry._load_detector.cache_clear()  # bỏ SlowDetector khỏi cache cho các test sau
