"""Test các hàm xử lý sự kiện của giao diện (gọi trực tiếp, không cần trình duyệt)."""

import os
import shutil
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image, ImageChops
from ultralytics.utils import ASSETS

from ui import handlers
from ui.handlers import SOURCE_UPLOAD, SOURCE_WEBCAM
from vision_count import LABEL_FULL, LABEL_NONE, HistoryStore
from vision_count.drawing import REGION_COLOR

BUS = str(ASSETS / "bus.jpg")
ZIDANE = str(ASSETS / "zidane.jpg")


@pytest.fixture
def history(tmp_path):
    return HistoryStore(tmp_path / "history.db")


def _single(history=None, model_key="nano", label_style=LABEL_FULL, tiled=False, region=None, classes=None):
    return handlers.count_single(
        SOURCE_UPLOAD, BUS, None, model_key, 0.25, classes or [], label_style, tiled, region, history
    )


# ---------- Một ảnh ----------

def test_tables_show_vietnamese_names():
    _, _, counts_table, detail_table, *_ = _single()
    assert "người (person)" in counts_table["Loại vật thể"].tolist()
    assert "xe buýt (bus)" in detail_table["Loại vật thể"].tolist()


def test_returns_three_download_files_in_separate_folders():
    first = _single()[4]
    second = _single()[4]
    assert [Path(p).suffix for p in first] == [".jpg", ".csv", ".csv"]
    assert all(Path(p).is_file() for p in first)
    assert Path(first[0]).parent != Path(second[0]).parent


def test_small_model_named_in_summary():
    _, summary, *_ = _single(model_key="small")
    assert "Small" in summary


def test_label_style_changes_image():
    full_img = _single(label_style=LABEL_FULL)[0]
    none_img = _single(label_style=LABEL_NONE)[0]
    assert ImageChops.difference(full_img, none_img).getbbox() is not None


def test_count_single_with_region_mentions_region():
    _, summary, counts_table, *_ = _single(region=(0, 0, 400, 1080))  # nửa trái ảnh xe buýt
    full = _single()[2]
    assert "vùng" in summary.lower()
    assert counts_table["Số lượng"].sum() < full["Số lượng"].sum()


def test_count_single_tiled_mentions_mode():
    _, summary, *_ = _single(tiled=True)
    assert "vật nhỏ" in summary.lower()


def test_count_single_saves_history(history):
    _single(history=history, tiled=True, region=(0, 0, 400, 1080))
    [entry] = history.recent()
    assert entry.source == "bus.jpg"
    assert entry.model_key == "nano"
    assert "Vật nhỏ" in entry.mode and "vùng" in entry.mode


def test_webcam_history_source_is_camera(history):
    frame = np.array(Image.open(ZIDANE).convert("RGB"))
    handlers.count_single(SOURCE_WEBCAM, None, frame, "nano", 0.25, [], LABEL_FULL, False, None, history)
    assert history.recent()[0].source == "Camera"


def test_missing_image_raises_friendly_error():
    with pytest.raises(gr.Error):
        handlers.count_single(SOURCE_UPLOAD, None, None, "nano", 0.25, [], LABEL_FULL, False, None, None)


# ---------- Chọn vùng bằng 2 lần bấm ----------

def _yellow(img, xy):
    return img.getpixel(xy) == REGION_COLOR


def test_count_single_returns_clean_base_and_resets_points():
    annotated, *_, base, points = _single(region=(0, 0, 400, 1080))
    assert points == []
    assert _yellow(annotated, (0, 500))  # ảnh hiển thị có khung vàng của vùng đang dùng
    assert not _yellow(base, (0, 500))  # ảnh nền để chọn vùng thì sạch, không có vùng cũ


def test_add_region_point_two_clicks_make_region():
    base = Image.new("RGB", (200, 200))
    marked, points, region, info = handlers.add_region_point(base, [], None, (150, 120))
    assert points == [(150, 120)] and region is None and "điểm thứ 2" in info
    assert _yellow(marked, (150, 120))
    marked2, points2, region2, info2 = handlers.add_region_point(base, points, None, (20, 30))
    assert points2 == [] and region2 == (20, 30, 150, 120)
    assert "Đếm" in info2


def test_new_region_replaces_old_marks_on_screen():
    base = Image.new("RGB", (200, 200))
    _, pts, _, _ = handlers.add_region_point(base, [], (10, 10, 100, 100), (120, 120))
    shown, _, region, _ = handlers.add_region_point(base, pts, (10, 10, 100, 100), (190, 190))
    assert region == (120, 120, 190, 190)
    assert _yellow(shown, (120, 150))  # cạnh vùng mới
    assert not _yellow(shown, (10, 50))  # không còn cạnh vùng cũ


def test_add_region_point_rejects_tiny_region():
    base = Image.new("RGB", (200, 200))
    shown, points, region, info = handlers.add_region_point(base, [(50, 50)], (1, 1, 100, 100), (52, 51))
    assert points == [] and region == (1, 1, 100, 100)  # giữ vùng cũ
    assert "quá nhỏ" in info
    assert not _yellow(shown, (50, 50))  # chấm của lần bấm hỏng không còn trên ảnh
    assert _yellow(shown, (1, 50))  # vùng đang dùng vẫn hiện


def test_add_region_point_without_image():
    out_img, points, region, info = handlers.add_region_point(None, [], None, (10, 10))
    assert out_img is None and points == [] and region is None
    assert "Đếm" in info


def test_clear_region_shows_clean_base():
    base = Image.new("RGB", (200, 200))
    shown, region, points, info = handlers.clear_region(base)
    assert region is None and points == [] and info == handlers.NO_REGION_TEXT
    assert ImageChops.difference(shown, base).getbbox() is None


# ---------- Nhiều ảnh ----------

def test_count_batch_summarizes_and_exports(history, tmp_path):
    bad = tmp_path / "hong.jpg"
    bad.write_text("khong phai anh")
    summary, table, gallery, csv_path = handlers.count_batch(
        [BUS, str(bad), ZIDANE], "nano", 0.25, [], LABEL_FULL, False, history
    )
    assert "3 ảnh" in summary and "1 lỗi" in summary
    assert table["Ảnh"].tolist() == ["bus.jpg", "hong.jpg", "zidane.jpg"]
    assert len(gallery) == 2  # chỉ các ảnh đếm được
    assert Path(csv_path).is_file()
    assert [e.source for e in history.recent()] == ["zidane.jpg", "bus.jpg"]


def test_count_batch_gallery_uses_small_previews(tmp_path):
    big = tmp_path / "lon.jpg"
    Image.open(BUS).convert("RGB").resize((2400, 3200)).save(big)
    _, _, gallery, _ = handlers.count_batch([str(big)], "nano", 0.25, [], LABEL_FULL, False, None)
    [(preview, caption)] = gallery
    assert max(preview.size) <= handlers.GALLERY_MAX_SIDE
    assert caption.startswith("lon.jpg")


def test_count_batch_without_files_raises():
    with pytest.raises(gr.Error):
        handlers.count_batch([], "nano", 0.25, [], LABEL_FULL, False, None)


# ---------- Lịch sử ----------

def test_history_table_and_clear(history):
    _single(history=history)
    table = handlers.history_table(history)
    assert table.columns.tolist() == handlers.HISTORY_COLUMNS
    assert table.iloc[0]["Nguồn"] == "bus.jpg"
    assert "người: 4" in table.iloc[0]["Chi tiết"]
    assert handlers.clear_history(history).empty
    assert history.recent() == []


# ---------- Dọn thư mục tạm (chuyển từ test_ui.py) ----------

def test_new_export_dir_removes_old_folders(tmp_path):
    old = tmp_path / "cu"
    old.mkdir()
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))
    recent = tmp_path / "moi"
    recent.mkdir()
    created = handlers.new_export_dir(root=tmp_path, max_age_seconds=3600)
    assert not old.exists() and recent.exists()
    assert created.parent == tmp_path and created.is_dir()


# ---------- Video ----------

def test_count_video_file(history, synthetic_video):
    summary, table, video_out, csv_path = handlers.count_video_file(
        str(synthetic_video), "nano", 0.25, [], 1, history
    )
    assert "3 vật thể khác nhau" in summary
    assert table.values.tolist() == [["người (person)", 3]]
    assert Path(video_out).is_file() and Path(csv_path).is_file()
    [entry] = history.recent()
    assert entry.source == "synthetic.mp4" and entry.total == 3 and entry.mode.startswith("Video")


def test_count_video_file_without_video_raises():
    with pytest.raises(gr.Error):
        handlers.count_video_file(None, "nano", 0.25, [], 1, None)


def test_count_video_file_unsupported_extension_raises(tmp_path, synthetic_video):
    clip = tmp_path / "clip.3gp"
    shutil.copy(synthetic_video, clip)
    with pytest.raises(gr.Error, match="chưa được hỗ trợ"):
        handlers.count_video_file(str(clip), "nano", 0.25, [], 1, None)


def test_count_video_file_shows_incomplete_warning(synthetic_video, monkeypatch):
    from vision_count import video as video_module

    monkeypatch.setattr(video_module, "_probe", lambda path, name=None: (15.0, 90))
    summary, *_ = handlers.count_video_file(str(synthetic_video), "nano", 0.25, [], 1, None)
    assert "Chỉ đọc được 30/90 khung hình" in summary


def test_count_video_file_bad_video_raises(tmp_path):
    bad = tmp_path / "hong.mp4"
    bad.write_text("x")
    with pytest.raises(gr.Error):
        handlers.count_video_file(str(bad), "nano", 0.25, [], 1, None)


# ---------- Ô lọc đổi theo model ----------

def test_class_choices_follow_model():
    update = handlers.class_choices("small")
    assert update["choices"][0] == ("người (person)", "person")
    assert len(update["choices"]) == 80
    assert update["value"] == []
