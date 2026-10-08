"""Test đếm video có tracking."""

import csv
import shutil

import cv2
import pytest
from ultralytics.utils import ASSETS

from vision_count import video as video_module
from vision_count import InvalidVideoError, VideoCountResult, count_video, get_detector, vi_label, write_video_csv


def test_counts_each_person_once(synthetic_video):
    result = count_video(get_detector(), synthetic_video)
    assert result.counts == {"person": 3}  # cộng từng khung sẽ ra ~79
    assert result.total == 3
    assert result.frames_processed == 30
    assert result.fps == 15
    assert result.duration_s == 2
    assert result.peak_in_frame == 3
    assert result.output_path is None


def test_writes_playable_annotated_video(synthetic_video, tmp_path):
    out = tmp_path / "ket_qua.mp4"
    result = count_video(get_detector(), synthetic_video, output_path=out, label_fn=vi_label)
    assert result.output_path == out and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode().lower()
    assert fourcc in {"avc1", "h264"}  # H.264: trình duyệt phát được


def test_vid_stride_processes_fewer_frames(synthetic_video):
    assert count_video(get_detector(), synthetic_video, vid_stride=2).frames_processed == 15


def test_short_tracks_are_ignored(synthetic_video):
    assert count_video(get_detector(), synthetic_video, min_track_frames=1000).total == 0


def test_progress_reports_every_frame(synthetic_video):
    calls = []
    count_video(get_detector(), synthetic_video, progress=lambda done, total: calls.append((done, total)))
    assert len(calls) == 30 and calls[-1] == (30, 30)


def test_invalid_video_raises(tmp_path):
    bad = tmp_path / "hong.mp4"
    bad.write_text("khong phai video")
    with pytest.raises(InvalidVideoError, match="không phải là file video"):
        count_video(get_detector(), bad)


def test_invalid_settings_raise(synthetic_video):
    with pytest.raises(ValueError):
        count_video(get_detector(), synthetic_video, confidence=2)
    with pytest.raises(ValueError):
        count_video(get_detector(), synthetic_video, vid_stride=0)


def test_image_detection_unchanged_after_video(synthetic_video):
    detector = get_detector()
    before = detector.detect(ASSETS / "bus.jpg").counts
    count_video(detector, synthetic_video)
    assert detector.detect(ASSETS / "bus.jpg").counts == before


def test_write_video_csv(tmp_path):
    result = VideoCountResult(counts={"person": 3, "car": 1}, frames_processed=30, fps=15, duration_s=2)
    path = write_video_csv(result, tmp_path / "video.csv")
    with open(path, encoding="utf-8-sig", newline="") as f:
        assert list(csv.reader(f)) == [
            ["Loại vật thể", "Tên gốc (model)", "Số vật khác nhau"],
            ["người", "person", "3"],
            ["ô tô", "car", "1"],
            ["Tổng cộng", "", "4"],
        ]


# ---------- Sửa sau review: định dạng lạ, file ảnh, video đọc thiếu ----------

def test_unsupported_extension_raises(tmp_path, synthetic_video):
    for name in ("clip.3gp", "khong_co_duoi"):
        path = tmp_path / name
        shutil.copy(synthetic_video, path)
        with pytest.raises(InvalidVideoError, match="chưa được hỗ trợ"):
            count_video(get_detector(), path)


def test_image_file_named_mp4_raises(tmp_path):
    fake = tmp_path / "anh_doi_ten.mp4"
    shutil.copy(ASSETS / "bus.jpg", fake)
    with pytest.raises(InvalidVideoError, match="ảnh"):
        count_video(get_detector(), fake)


def test_error_message_uses_given_name(tmp_path):
    bad = tmp_path / "video.mp4"
    bad.write_text("x")
    with pytest.raises(InvalidVideoError, match="clip_cua_toi.mp4"):
        count_video(get_detector(), bad, name="clip_cua_toi.mp4")


def test_complete_video_has_no_warning(synthetic_video):
    result = count_video(get_detector(), synthetic_video)
    assert result.frames_expected == 30 and result.complete and result.warning is None


def test_incomplete_read_warns(synthetic_video, monkeypatch):
    # Giả lập video hỏng giữa chừng: phần đầu file khai báo 90 khung nhưng chỉ đọc được 30
    monkeypatch.setattr(video_module, "_probe", lambda path, name=None: (15.0, 90))
    result = count_video(get_detector(), synthetic_video)
    assert result.frames_processed == 30 and result.frames_expected == 90
    assert not result.complete
    assert "Chỉ đọc được 30/90 khung hình" in result.warning


# ---------- Đợt sửa điểm nhỏ ----------

def test_peak_counts_only_counted_tracks(synthetic_video):
    result = count_video(get_detector(), synthetic_video, min_track_frames=1000)
    assert result.total == 0 and result.peak_in_frame == 0


def test_min_frames_scale_with_stride():
    assert video_module._effective_min_frames(3, 1) == 3
    assert video_module._effective_min_frames(3, 2) == 2
    assert video_module._effective_min_frames(3, 5) == 2  # vẫn bỏ ID chỉ thấy 1 khung
    assert video_module._effective_min_frames(1, 5) == 1
    assert video_module._effective_min_frames(1000, 1) == 1000


def test_tracker_config_follows_low_confidence():
    import yaml

    low = yaml.safe_load(open(video_module._tracker_config(0.1), encoding="utf-8"))
    assert low["track_high_thresh"] == 0.1 and low["new_track_thresh"] == 0.1
    assert low["track_low_thresh"] <= 0.1
    default = yaml.safe_load(open(video_module._tracker_config(0.5), encoding="utf-8"))
    assert default["track_high_thresh"] == 0.25 and default["new_track_thresh"] == 0.25
    assert default["tracker_type"] == "bytetrack"


def test_low_confidence_still_counts(synthetic_video):
    assert count_video(get_detector(), synthetic_video, confidence=0.1).counts.get("person", 0) >= 3
