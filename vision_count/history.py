"""Lịch sử đếm lưu trong SQLite (có sẵn trong Python, không cần cài thêm).

Chỉ lưu số liệu (thời gian, nguồn ảnh, model, chế độ, số lượng), KHÔNG lưu ảnh.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from vision_count import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    model_key TEXT NOT NULL,
    mode TEXT NOT NULL,
    total INTEGER NOT NULL,
    counts_json TEXT NOT NULL
)
"""


class CountResult(Protocol):
    """Bất kỳ kết quả nào có số đếm theo loại và tổng (ảnh: DetectionResult, video: VideoCountResult)."""

    counts: dict[str, int]

    @property
    def total(self) -> int: ...


@dataclass
class HistoryEntry:
    id: int
    created_at: str  # "YYYY-MM-DD HH:MM:SS"
    source: str  # tên file ảnh, hoặc "Camera"
    model_key: str
    mode: str  # mô tả chế độ, ví dụ "Thường", "Vật nhỏ, trong vùng đã chọn"
    total: int
    counts: dict[str, int]


class HistoryStore:
    def __init__(self, db_path: str | Path = config.HISTORY_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> _Connection:
        # Mỗi lần dùng mở một kết nối mới rồi đóng ngay: đơn giản và an toàn khi Gradio chạy nhiều luồng
        return _Connection(self.db_path)

    def add(
        self, source: str, model_key: str, mode: str, result: CountResult, created_at: datetime | None = None
    ) -> int:
        """Lưu một lần đếm, trả về id của dòng mới."""
        stamp = (created_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO history (created_at, source, model_key, mode, total, counts_json) VALUES (?, ?, ?, ?, ?, ?)",
                (stamp, source, model_key, mode, result.total, json.dumps(result.counts, ensure_ascii=False)),
            )
            return cursor.lastrowid

    def recent(self, limit: int = 50) -> list[HistoryEntry]:
        """Các lần đếm gần nhất, mới nhất đứng đầu."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, source, model_key, mode, total, counts_json "
                "FROM history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [HistoryEntry(*row[:6], counts=json.loads(row[6])) for row in rows]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM history")


class _Connection:
    """Mở kết nối SQLite, commit khi xong (rollback nếu lỗi) và luôn đóng kết nối."""

    def __init__(self, path: Path):
        self._conn = sqlite3.connect(path)

    def __enter__(self) -> sqlite3.Connection:
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        with closing(self._conn):
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
