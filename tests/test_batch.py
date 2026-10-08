"""Test đếm nhiều ảnh."""

import csv

from ultralytics.utils import ASSETS

from vision_count import BatchItem, Detection, DetectionResult, count_many, get_detector, write_batch_csv


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def test_count_many_keeps_going_after_bad_file(tmp_path):
    bad = tmp_path / "hong.jpg"
    bad.write_text("khong phai anh")
    items = count_many(get_detector(), [ASSETS / "bus.jpg", bad, ASSETS / "zidane.jpg"])
    assert [i.name for i in items] == ["bus.jpg", "hong.jpg", "zidane.jpg"]
    assert items[0].result.counts["person"] >= 3
    assert items[1].result is None and "không phải là file ảnh" in items[1].error
    assert items[2].result.total >= 2


def test_write_batch_csv_has_columns_per_class_and_totals(tmp_path):
    def res(counts):
        dets = [Detection(label, 0, 0.9, (0.0, 0.0, 1.0, 1.0)) for label, n in counts.items() for _ in range(n)]
        return DetectionResult(detections=dets, counts=counts)

    items = [
        BatchItem("a.jpg", res({"person": 2, "car": 1})),
        BatchItem("b.jpg", res({"person": 3})),
        BatchItem("c.jpg", error="'c.jpg' không phải là file ảnh hợp lệ"),
    ]
    path = write_batch_csv(items, tmp_path / "tong_hop.csv")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert _read(path) == [
        ["Ảnh", "Tổng", "người", "ô tô", "Lỗi"],
        ["a.jpg", "3", "2", "1", ""],
        ["b.jpg", "3", "3", "0", ""],
        ["c.jpg", "", "", "", "'c.jpg' không phải là file ảnh hợp lệ"],
        ["Tổng cộng", "6", "5", "1", ""],
    ]
