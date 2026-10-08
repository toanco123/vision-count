"""Test xuất kết quả ra file ảnh + CSV."""

import csv
from datetime import datetime

from PIL import Image

from vision_count import Detection, DetectionResult, export_result

FIXED_TIME = datetime(2026, 10, 8, 15, 30, 45)


def _result():
    dets = [
        Detection("person", 0, 0.91, (1.0, 2.0, 30.0, 40.0)),
        Detection("person", 0, 0.80, (5.0, 6.0, 50.0, 60.0)),
        Detection("bus", 5, 0.95, (0.0, 0.0, 100.0, 80.0)),
    ]
    return DetectionResult(detections=dets, counts={"person": 2, "bus": 1}, image_width=100, image_height=80)


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def test_export_creates_three_files(tmp_path):
    files = export_result(Image.new("RGB", (100, 80)), _result(), tmp_path, FIXED_TIME)
    assert [p.suffix for p in files] == [".jpg", ".csv", ".csv"]
    assert all(p.is_file() and p.parent == tmp_path for p in files)
    assert Image.open(files[0]).size == (100, 80)


def test_export_file_names_have_timestamp(tmp_path):
    files = export_result(Image.new("RGB", (100, 80)), _result(), tmp_path, FIXED_TIME)
    assert [p.name for p in files] == [
        "ket_qua_20261008_153045.jpg",
        "so_luong_20261008_153045.csv",
        "chi_tiet_20261008_153045.csv",
    ]


def test_counts_csv_has_vietnamese_names_and_total(tmp_path):
    _, counts_csv, _ = export_result(Image.new("RGB", (100, 80)), _result(), tmp_path, FIXED_TIME)
    assert _read_csv(counts_csv) == [
        ["Loại vật thể", "Tên gốc (model)", "Số lượng"],
        ["người", "person", "2"],
        ["xe buýt", "bus", "1"],
        ["Tổng cộng", "", "3"],
    ]


def test_detail_csv_lists_every_box(tmp_path):
    _, _, detail_csv = export_result(Image.new("RGB", (100, 80)), _result(), tmp_path, FIXED_TIME)
    rows = _read_csv(detail_csv)
    assert rows[0] == ["#", "Loại vật thể", "Tên gốc (model)", "Độ tin cậy", "x1", "y1", "x2", "y2"]
    assert rows[1] == ["1", "người", "person", "0.91", "1.0", "2.0", "30.0", "40.0"]
    assert len(rows) == 4


def test_csv_files_have_utf8_bom(tmp_path):
    # BOM giúp Excel nhận đúng UTF-8, không bị vỡ dấu tiếng Việt
    _, counts_csv, detail_csv = export_result(Image.new("RGB", (100, 80)), _result(), tmp_path, FIXED_TIME)
    for path in (counts_csv, detail_csv):
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_export_empty_result(tmp_path):
    files = export_result(Image.new("RGB", (100, 80)), DetectionResult(), tmp_path, FIXED_TIME)
    assert len(files) == 3
    assert _read_csv(files[1]) == [["Loại vật thể", "Tên gốc (model)", "Số lượng"], ["Tổng cộng", "", "0"]]
    assert len(_read_csv(files[2])) == 1  # chỉ có dòng tiêu đề


def test_export_same_second_does_not_overwrite(tmp_path):
    first = export_result(Image.new("RGB", (100, 80)), _result(), tmp_path, FIXED_TIME)
    second = export_result(Image.new("RGB", (100, 80)), _result(), tmp_path, FIXED_TIME)
    assert [p.name for p in second] == [
        "ket_qua_20261008_153045_2.jpg",
        "so_luong_20261008_153045_2.csv",
        "chi_tiet_20261008_153045_2.csv",
    ]
    assert all(p.is_file() for p in first + second)
