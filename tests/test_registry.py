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
