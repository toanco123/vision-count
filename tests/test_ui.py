"""Test hàm xử lý nút 'Đếm' của giao diện Gradio (gọi thẳng, không cần mở trình duyệt)."""

from pathlib import Path

import pytest
from ultralytics.utils import ASSETS

from ui.gradio_app import SOURCE_UPLOAD, build_app
from vision_count import ObjectDetector

BUS_IMAGE = str(ASSETS / "bus.jpg")


@pytest.fixture(scope="module")
def count_objects():
    app = build_app(ObjectDetector())
    # Lấy hàm được gắn vào nút "Đếm" trong app
    return next(f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == "count_objects")


def test_tables_show_vietnamese_names(count_objects):
    _, _, counts_table, detail_table, _ = count_objects(SOURCE_UPLOAD, BUS_IMAGE, None, 0.25, [])
    assert "người (person)" in counts_table["Loại vật thể"].tolist()
    assert "xe buýt (bus)" in detail_table["Loại vật thể"].tolist()


def test_returns_three_download_files(count_objects):
    *_, files = count_objects(SOURCE_UPLOAD, BUS_IMAGE, None, 0.25, [])
    assert [Path(p).suffix for p in files] == [".jpg", ".csv", ".csv"]
    assert all(Path(p).is_file() for p in files)


def test_each_run_uses_separate_folder(count_objects):
    *_, first = count_objects(SOURCE_UPLOAD, BUS_IMAGE, None, 0.25, [])
    *_, second = count_objects(SOURCE_UPLOAD, BUS_IMAGE, None, 0.25, [])
    assert Path(first[0]).parent != Path(second[0]).parent
