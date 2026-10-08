"""Test hàm xử lý nút 'Đếm' của giao diện Gradio (gọi thẳng, không cần mở trình duyệt)."""

import os
import time
from pathlib import Path

import pytest
from PIL import ImageChops
from ultralytics.utils import ASSETS

from ui.gradio_app import SOURCE_UPLOAD, build_app, new_export_dir
from vision_count import LABEL_FULL, LABEL_NONE

BUS_IMAGE = str(ASSETS / "bus.jpg")


@pytest.fixture(scope="module")
def count_objects():
    app = build_app()
    # Lấy hàm được gắn vào nút "Đếm" trong app
    return next(f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == "count_objects")


def _run(fn, model_key="nano", label_style=LABEL_FULL, classes=None):
    return fn(SOURCE_UPLOAD, BUS_IMAGE, None, model_key, 0.25, classes or [], label_style)


def test_tables_show_vietnamese_names(count_objects):
    _, _, counts_table, detail_table, _ = _run(count_objects)
    assert "người (person)" in counts_table["Loại vật thể"].tolist()
    assert "xe buýt (bus)" in detail_table["Loại vật thể"].tolist()


def test_returns_three_download_files(count_objects):
    *_, files = _run(count_objects)
    assert [Path(p).suffix for p in files] == [".jpg", ".csv", ".csv"]
    assert all(Path(p).is_file() for p in files)


def test_each_run_uses_separate_folder(count_objects):
    *_, first = _run(count_objects)
    *_, second = _run(count_objects)
    assert Path(first[0]).parent != Path(second[0]).parent


def test_small_model_is_used_and_named_in_summary(count_objects):
    _, summary, counts_table, _, _ = _run(count_objects, model_key="small")
    assert "Small" in summary
    assert "người (person)" in counts_table["Loại vật thể"].tolist()


def test_label_style_changes_image(count_objects):
    full_img = _run(count_objects, label_style=LABEL_FULL)[0]
    none_img = _run(count_objects, label_style=LABEL_NONE)[0]
    assert ImageChops.difference(full_img, none_img).getbbox() is not None


def test_new_export_dir_removes_old_folders(tmp_path):
    old = tmp_path / "cu"
    old.mkdir()
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))
    recent = tmp_path / "moi"
    recent.mkdir()

    created = new_export_dir(root=tmp_path, max_age_seconds=3600)

    assert not old.exists()
    assert recent.exists()
    assert created.parent == tmp_path and created.is_dir()
