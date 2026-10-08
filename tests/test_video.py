"""Test đếm video có tracking."""

import csv

import cv2
import pytest
from ultralytics.utils import ASSETS

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
