"""Test dựng giao diện Gradio (các hàm xử lý được test ở test_handlers.py)."""

from vision_count import HistoryStore

from ui.gradio_app import build_app


def test_uploaded_files_are_kept_for_a_day(tmp_path):
    # Gradio xóa cả ảnh người dùng đã tải lên theo delete_cache; 1 giờ là quá ngắn khi tab vẫn mở
    _, max_age = build_app(HistoryStore(tmp_path / "h.db")).delete_cache
    assert max_age >= 86400


def test_app_has_three_main_tabs(tmp_path):
    app = build_app(HistoryStore(tmp_path / "h.db"))
    labels = {block.label for block in app.blocks.values() if type(block).__name__ == "Tab"}
    assert {"Một ảnh", "Nhiều ảnh", "Lịch sử"} <= labels


def test_result_image_is_not_an_upload_box(tmp_path):
    # Ảnh kết quả được dùng làm đầu vào của sự kiện bấm chọn vùng; nếu không khóa lại,
    # Gradio tự biến nó thành ô tải ảnh lên (có nút upload/webcam), gây rối cho người dùng
    app = build_app(HistoryStore(tmp_path / "h.db"))
    [result_image] = [
        b for b in app.blocks.values() if type(b).__name__ == "Image" and str(b.label).startswith("Kết quả")
    ]
    assert result_image.interactive is False
