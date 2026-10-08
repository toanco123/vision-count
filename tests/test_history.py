"""Test lịch sử đếm (SQLite)."""

from datetime import datetime

from vision_count import Detection, DetectionResult, HistoryStore


def _result(counts):
    dets = [Detection(label, 0, 0.9, (0.0, 0.0, 1.0, 1.0)) for label, n in counts.items() for _ in range(n)]
    return DetectionResult(detections=dets, counts=counts)


def test_add_and_read_back(tmp_path):
    store = HistoryStore(tmp_path / "sub" / "history.db")  # thư mục con chưa có: phải tự tạo
    new_id = store.add("bus.jpg", "nano", "Thường", _result({"person": 4, "bus": 1}), datetime(2026, 10, 8, 9, 30))
    [entry] = store.recent()
    assert entry.id == new_id
    assert entry.created_at == "2026-10-08 09:30:00"
    assert (entry.source, entry.model_key, entry.mode, entry.total) == ("bus.jpg", "nano", "Thường", 5)
    assert entry.counts == {"person": 4, "bus": 1}


def test_recent_is_newest_first_and_limited(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    for i in range(5):
        store.add(f"anh{i}.jpg", "nano", "Thường", _result({"person": i}))
    assert [e.source for e in store.recent(limit=3)] == ["anh4.jpg", "anh3.jpg", "anh2.jpg"]


def test_history_survives_new_store_instance(tmp_path):
    HistoryStore(tmp_path / "history.db").add("a.jpg", "small", "Vật nhỏ", _result({"car": 2}))
    assert HistoryStore(tmp_path / "history.db").recent()[0].counts == {"car": 2}


def test_clear_removes_everything(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    store.add("a.jpg", "nano", "Thường", _result({}))
    store.clear()
    assert store.recent() == []
