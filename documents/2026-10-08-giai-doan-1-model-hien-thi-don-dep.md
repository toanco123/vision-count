# Giai đoạn 1: Chọn model, kiểu nhãn gọn, dọn dẹp: Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Mục tiêu:** Người dùng chọn được model nano hoặc small và chọn kiểu nhãn trên ảnh (đầy đủ / chỉ số thứ tự / chỉ khung). Đồng thời dọn các điểm nhỏ còn tồn từ lần review trước: file tạm không tự xóa, `export_result` ghi đè khi gọi trong cùng một giây, test font phụ thuộc máy, tìm font lặp lại theo từng cỡ chữ.

**Kiến trúc:** Danh sách model nằm trong `config.py`. Module mới `registry.py` nạp mỗi model một lần rồi giữ lại (`get_detector`), để cả Gradio lẫn FastAPI sau này cùng dùng. Kiểu nhãn là một tham số mới của `draw_detections`. Việc dọn file tạm thuộc về lớp giao diện (`ui/gradio_app.py`).

**Công nghệ:** Python 3.12, Ultralytics (`yolo11n.pt`, `yolo11s.pt`), Pillow, Gradio 6, pytest.

**Spec:** Không có file spec riêng. Yêu cầu là nhóm 1 (mục 1, 2, 3) trong danh sách đề xuất ngày 2026-10-08, người dùng đồng ý "thực hiện từng giai đoạn":
1. Chọn model nano hoặc small ngay trên giao diện.
2. Chế độ hiển thị gọn khi vật dày đặc: chỉ số thứ tự, hoặc chỉ khung.
3. Dọn các điểm nhỏ còn tồn từ lần review trước.

## Ràng buộc chung

- Chạy offline. Chỉ cần mạng **một lần** để tải `yolo11s.pt` (khoảng 19MB) khi chọn small lần đầu.
- Không thêm thư viện vào `requirements.txt`.
- `ObjectDetector.detect()` giữ nguyên hành vi và chữ ký.
- Chú thích code bằng tiếng Việt, theo phong cách file hiện có.

## Ngoài phạm vi (đã cân nhắc)

- **Ô lọc gõ không dấu ("nguoi" ra "người"):** `gr.Dropdown` của Gradio không cho tự viết hàm lọc. Muốn làm thì phải nhét thêm chữ không dấu vào nhãn hiển thị, làm rối giao diện. Gõ tên tiếng Anh ("person") vẫn tìm được, nên README sẽ ghi mẹo này.

## Điểm cần chú ý khi review

1. **Chọn small lần đầu khi mất mạng:** việc tải model thất bại phải hiện thông báo lỗi trên giao diện, không làm treo hay sập app. Test: `test_unknown_model_key_raises` chỉ phủ trường hợp sai khóa; trường hợp mất mạng cần reviewer kiểm tra đường lỗi trong `count_objects`.
2. **Đổi model qua lại nhiều lần:** không được nạp lại model mỗi lần bấm, sẽ rất chậm. Test: `test_get_detector_caches_instance`.
3. **Máy không có font tiếng Việt:** vẫn vẽ được và không lỗi. Test: `test_find_font_falls_back_without_vietnamese_font`.
4. **Gọi `export_result` nhiều lần trong cùng một giây vào cùng thư mục:** không mất file cũ. Test: `test_export_same_second_does_not_overwrite`.
5. **Thư mục tạm phình dần sau nhiều ngày dùng:** thư mục cũ hơn 1 giờ phải bị xóa ở lần đếm kế tiếp. Test: `test_new_export_dir_removes_old_folders`.

---

## Cấu trúc file

| File | Việc | Trách nhiệm |
|---|---|---|
| `vision_count/config.py` | Sửa | Thêm `AVAILABLE_MODELS`, `DEFAULT_MODEL_KEY` |
| `vision_count/detector.py` | Sửa | Lưu `self.model_path` (chỉ thêm thuộc tính) |
| `vision_count/registry.py` | Tạo mới | `get_detector()`, `is_model_downloaded()` |
| `vision_count/drawing.py` | Sửa | Tham số `label_style`, tìm font một lần |
| `vision_count/export.py` | Sửa | Thêm hậu tố `_2`, `_3`... khi trùng tên |
| `vision_count/__init__.py` | Sửa | Xuất thêm các tên mới |
| `ui/gradio_app.py` | Sửa | Ô chọn model, ô chọn kiểu nhãn, dọn thư mục tạm |
| `app.py` | Sửa | Nạp sẵn model mặc định qua `get_detector()` |
| `tests/test_registry.py` | Tạo mới | Test chọn model |
| `tests/test_drawing.py`, `tests/test_export.py`, `tests/test_ui.py` | Sửa | Test mới, cập nhật chữ ký |
| `README.md` | Sửa | Mô tả tính năng mới |

---

### Task 1: Danh sách model và `get_detector`

**Files:**
- Sửa: `vision_count/config.py`, `vision_count/detector.py`, `vision_count/__init__.py`
- Tạo: `vision_count/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces:
  - `config.AVAILABLE_MODELS: dict[str, tuple[str, str]]`, ánh xạ khóa → (tên hiển thị, file weights).
  - `config.DEFAULT_MODEL_KEY = "nano"`.
  - `get_detector(model_key: str = "nano") -> ObjectDetector`: cache mỗi khóa một lần; khóa sai thì ném `ValueError`.
  - `is_model_downloaded(model_key: str) -> bool`.
  - `ObjectDetector.model_path: Path`.

- [ ] **Bước 1: Viết test (sẽ fail)**: tạo `tests/test_registry.py`

```python
"""Test chọn model qua get_detector."""

import pytest

from vision_count import config, get_detector, is_model_downloaded


def test_get_detector_caches_instance():
    # Đổi model qua lại không được nạp lại từ đầu (rất chậm)
    assert get_detector("nano") is get_detector("nano")


def test_small_model_uses_its_own_weights():
    detector = get_detector("small")  # lần đầu sẽ tải yolo11s.pt (~19MB)
    assert detector.model_path.name == "yolo11s.pt"
    assert detector is not get_detector("nano")
    assert len(detector.class_names) == 80


def test_unknown_model_key_raises():
    with pytest.raises(ValueError):
        get_detector("khong_ton_tai")


def test_is_model_downloaded():
    get_detector("nano")
    assert is_model_downloaded("nano") is True
    assert is_model_downloaded("khong_ton_tai") is False


def test_default_key_is_in_available_models():
    assert config.DEFAULT_MODEL_KEY in config.AVAILABLE_MODELS
```

- [ ] **Bước 2: Chạy test, phải FAIL**

Chạy: `.venv/bin/pytest tests/test_registry.py -v`
Kỳ vọng: `ImportError: cannot import name 'get_detector'`.

- [ ] **Bước 3: Viết code**

Thêm vào cuối `vision_count/config.py`:

```python
# Các model cho người dùng chọn: khóa -> (tên hiển thị, file weights).
# Bản small chính xác hơn nhưng chậm hơn; lần đầu chọn sẽ tải file (~19MB).
AVAILABLE_MODELS = {
    "nano": ("Nano: nhanh nhất", "yolo11n.pt"),
    "small": ("Small: chính xác hơn, chậm hơn", "yolo11s.pt"),
}
DEFAULT_MODEL_KEY = "nano"
```

Trong `ObjectDetector.__init__` (`vision_count/detector.py`), lưu đường dẫn đã resolve:

```python
        self.model_path = self._resolve_model_path(model_path)
        self.model = YOLO(str(self.model_path))
```

Tạo `vision_count/registry.py`:

```python
"""Nạp và giữ lại các model, để đổi model qua lại không phải nạp lại từ đầu."""

from __future__ import annotations

from functools import lru_cache

from vision_count import config
from vision_count.detector import ObjectDetector


@lru_cache(maxsize=None)
def get_detector(model_key: str = config.DEFAULT_MODEL_KEY) -> ObjectDetector:
    """Trả về detector của model được chọn. Mỗi model chỉ nạp một lần."""
    if model_key not in config.AVAILABLE_MODELS:
        raise ValueError(f"Không có model '{model_key}'. Chọn một trong: {', '.join(config.AVAILABLE_MODELS)}")
    _, weights = config.AVAILABLE_MODELS[model_key]
    return ObjectDetector(weights)


def is_model_downloaded(model_key: str) -> bool:
    """Model đã có sẵn trong thư mục models/ chưa (chưa có thì lần đầu dùng sẽ phải tải)."""
    if model_key not in config.AVAILABLE_MODELS:
        return False
    _, weights = config.AVAILABLE_MODELS[model_key]
    return (config.MODELS_DIR / weights).is_file()
```

Trong `vision_count/__init__.py`, thêm `from vision_count.registry import get_detector, is_model_downloaded` và thêm hai tên này vào `__all__`.

- [ ] **Bước 4: Chạy toàn bộ test, phải PASS**

Chạy: `.venv/bin/pytest -q`
Kỳ vọng: tất cả passed (lần đầu sẽ tải `yolo11s.pt`).

- [ ] **Bước 5: Commit**: `git commit -m "Thêm danh sách model và get_detector để chọn nano/small"`

---

### Task 2: Kiểu nhãn trên ảnh và font

**Files:**
- Sửa: `vision_count/drawing.py`, `vision_count/__init__.py`
- Test: `tests/test_drawing.py`

**Interfaces:**
- Produces:
  - `LABEL_FULL = "full"`, `LABEL_NUMBER = "number"`, `LABEL_NONE = "none"`, `LABEL_STYLES = (LABEL_FULL, LABEL_NUMBER, LABEL_NONE)`.
  - `draw_detections(image, detections, label_fn=None, label_style=LABEL_FULL)`. Kiểu nhãn sai thì ném `ValueError`.
  - `find_font(size)` giữ nguyên chữ ký.

- [ ] **Bước 1: Viết test (sẽ fail)**

Ghi đè `tests/test_drawing.py`:

```python
"""Test phần vẽ khung và nhãn."""

import pytest
from PIL import Image, ImageChops, ImageDraw

from vision_count import LABEL_FULL, LABEL_NONE, LABEL_NUMBER, Detection, draw_detections, vi_label
from vision_count import drawing
from vision_count.drawing import find_font


def _sample_detection():
    return Detection(label="person", class_id=0, confidence=0.9, box=(10, 30, 120, 180))


def _draw(style, label_fn=None):
    return draw_detections(Image.new("RGB", (200, 200)), [_sample_detection()], label_fn=label_fn, label_style=style)


@pytest.fixture
def clear_font_cache():
    drawing._find_font_name.cache_clear()
    find_font.cache_clear()
    yield
    drawing._find_font_name.cache_clear()
    find_font.cache_clear()


@pytest.mark.skipif(drawing._find_font_name() is None, reason="Máy không có font hỗ trợ tiếng Việt")
def test_find_font_uses_a_real_font_file():
    # Font thật có đường dẫn file (str); font mặc định của Pillow nằm trong bộ nhớ (không có dấu tiếng Việt)
    assert isinstance(find_font(20).path, str)


def test_find_font_falls_back_without_vietnamese_font(monkeypatch, clear_font_cache):
    monkeypatch.setattr(drawing, "VIETNAMESE_FONTS", ["khong_co_font_nay.ttf"])
    font = find_font(20)
    assert not isinstance(font.path, str)  # đã dùng font mặc định
    _draw(LABEL_FULL, vi_label)  # vẫn vẽ được, không lỗi


def test_label_fn_is_used_for_full_label():
    seen = []
    _draw(LABEL_FULL, lambda label: seen.append(label) or vi_label(label))
    assert seen == ["person"]


def test_number_style_does_not_need_name():
    seen = []
    _draw(LABEL_NUMBER, lambda label: seen.append(label) or label)
    assert seen == []


def test_styles_draw_different_images():
    full, number, none = _draw(LABEL_FULL), _draw(LABEL_NUMBER), _draw(LABEL_NONE)
    assert ImageChops.difference(full, number).getbbox() is not None
    assert ImageChops.difference(number, none).getbbox() is not None


def test_none_style_draws_only_box():
    # Ảnh 200px: độ dày nét = max(2, round(3 * 0.2)) = 2; class_id 0 dùng màu PALETTE[0]
    expected = Image.new("RGB", (200, 200))
    ImageDraw.Draw(expected).rectangle((10, 30, 120, 180), outline=drawing.PALETTE[0], width=2)
    assert ImageChops.difference(_draw(LABEL_NONE), expected).getbbox() is None


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        _draw("khong_co")


def test_default_draws_without_label_fn():
    out = draw_detections(Image.new("RGB", (200, 200)), [_sample_detection()])
    assert out.size == (200, 200)
```

- [ ] **Bước 2: Chạy test, phải FAIL**

Chạy: `.venv/bin/pytest tests/test_drawing.py -v`
Kỳ vọng: `ImportError: cannot import name 'LABEL_FULL'`.

- [ ] **Bước 3: Sửa `vision_count/drawing.py`**

Thay hàm `find_font` bằng hai hàm sau. Tìm tên font một lần, rồi cache font theo từng cỡ chữ:

```python
@lru_cache(maxsize=1)
def _find_font_name() -> str | None:
    """Tìm (một lần) tên font hỗ trợ tiếng Việt có trên máy; không có thì trả về None."""
    for name in VIETNAMESE_FONTS:
        try:
            ImageFont.truetype(name, 10)
            return name
        except OSError:
            continue
    return None


@lru_cache(maxsize=16)
def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Font hỗ trợ tiếng Việt ở cỡ chữ size; không có thì dùng font mặc định (mất dấu nhưng không lỗi)."""
    name = _find_font_name()
    return ImageFont.truetype(name, size) if name else ImageFont.load_default(size=size)
```

Thêm các hằng số kiểu nhãn ngay trên `draw_detections`:

```python
# Kiểu nhãn trên ảnh. Khi vật dày đặc, nhãn đầy đủ sẽ đè lên nhau, nên có thêm 2 kiểu gọn.
LABEL_FULL = "full"  # "#1 người 87%"
LABEL_NUMBER = "number"  # "1"
LABEL_NONE = "none"  # chỉ vẽ khung
LABEL_STYLES = (LABEL_FULL, LABEL_NUMBER, LABEL_NONE)
```

Thêm tham số `label_style: str = LABEL_FULL` vào `draw_detections`, ghi vào docstring, kiểm tra ở đầu hàm:

```python
    if label_style not in LABEL_STYLES:
        raise ValueError(f"Kiểu nhãn không hợp lệ: {label_style}. Chọn một trong: {', '.join(LABEL_STYLES)}")
```

Trong vòng lặp, ngay sau `draw.rectangle(...)`:

```python
        if label_style == LABEL_NONE:
            continue
        if label_style == LABEL_NUMBER:
            text = str(index)
        else:
            # Nhãn dạng "#1 người 87%": số thứ tự giúp đối chiếu khi đếm bằng mắt
            name = label_fn(det.label) if label_fn else det.label
            text = f"#{index} {name} {det.confidence:.0%}"
```

Trong `vision_count/__init__.py`, xuất thêm `LABEL_FULL`, `LABEL_NUMBER`, `LABEL_NONE`, `LABEL_STYLES` từ `vision_count.drawing`.

- [ ] **Bước 4: Chạy toàn bộ test, phải PASS**: `.venv/bin/pytest -q`

- [ ] **Bước 5: Commit**: `git commit -m "Thêm kiểu nhãn gọn (chỉ số thứ tự / chỉ khung), tìm font một lần"`

---

### Task 3: `export_result` không ghi đè file cũ

**Files:**
- Sửa: `vision_count/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Produces: `export_result(...)` giữ nguyên chữ ký. Khi trùng tên, các file thêm hậu tố `_2`, `_3`..., ví dụ `ket_qua_20261008_153045_2.jpg`.

- [ ] **Bước 1: Thêm test vào `tests/test_export.py` (sẽ fail)**

```python
def test_export_same_second_does_not_overwrite(tmp_path):
    first = export_result(Image.new("RGB", (100, 80)), _result(), tmp_path, FIXED_TIME)
    second = export_result(Image.new("RGB", (100, 80)), _result(), tmp_path, FIXED_TIME)
    assert [p.name for p in second] == [
        "ket_qua_20261008_153045_2.jpg",
        "so_luong_20261008_153045_2.csv",
        "chi_tiet_20261008_153045_2.csv",
    ]
    assert all(p.is_file() for p in first + second)
```

- [ ] **Bước 2: Chạy test, phải FAIL**

Chạy: `.venv/bin/pytest tests/test_export.py -v`
Kỳ vọng: `test_export_same_second_does_not_overwrite` FAIL (tên file không có `_2`).

- [ ] **Bước 3: Sửa `vision_count/export.py`**

Thêm hằng số dưới `DETAIL_HEADER`:

```python
# (tiền tố, đuôi) của 3 file xuất ra, theo đúng thứ tự trả về
EXPORT_FILES = [("ket_qua", ".jpg"), ("so_luong", ".csv"), ("chi_tiet", ".csv")]
```

Thay đoạn tính `stamp` và 3 đường dẫn:

```python
    # Dấu thời gian trong tên file để các lần tải về không trùng tên nhau
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    # Nếu đã có file cùng tên (gọi 2 lần trong cùng 1 giây) thì thêm hậu tố _2, _3...
    base, n = stamp, 1
    while any((out_dir / f"{prefix}_{stamp}{ext}").exists() for prefix, ext in EXPORT_FILES):
        n += 1
        stamp = f"{base}_{n}"
    image_path, counts_path, detail_path = (out_dir / f"{prefix}_{stamp}{ext}" for prefix, ext in EXPORT_FILES)
```

Xóa 3 dòng gán `image_path = ...`, `counts_path = ...`, `detail_path = ...` cũ.

- [ ] **Bước 4: Chạy toàn bộ test, phải PASS**: `.venv/bin/pytest -q`

- [ ] **Bước 5: Commit**: `git commit -m "export_result: thêm hậu tố khi trùng tên, không ghi đè file cũ"`

---

### Task 4: Giao diện: chọn model, kiểu nhãn, dọn thư mục tạm; README

**Files:**
- Sửa: `ui/gradio_app.py`, `app.py`, `README.md`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `get_detector`, `is_model_downloaded`, `config.AVAILABLE_MODELS`, `config.DEFAULT_MODEL_KEY` (Task 1); `LABEL_FULL`, `LABEL_NUMBER`, `LABEL_NONE`, `draw_detections(..., label_style=)` (Task 2).
- Produces:
  - `build_app() -> gr.Blocks`, không còn tham số `detector`.
  - Hàm xử lý `count_objects(source_mode, image_path, webcam_image, model_key, confidence, selected_classes, label_style)` trả về 5 giá trị như cũ.
  - `new_export_dir(root: Path = EXPORT_ROOT, max_age_seconds: int = 3600) -> Path`.

- [ ] **Bước 1: Ghi đè `tests/test_ui.py` (sẽ fail)**

```python
"""Test hàm xử lý nút 'Đếm' của giao diện Gradio (gọi thẳng, không cần mở trình duyệt)."""

import os
import time
from pathlib import Path

import pytest
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
    assert list(full_img.getdata()) != list(none_img.getdata())


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
```

- [ ] **Bước 2: Chạy test, phải FAIL**

Chạy: `.venv/bin/pytest tests/test_ui.py -v`
Kỳ vọng: `ImportError: cannot import name 'new_export_dir'`.

- [ ] **Bước 3: Sửa `ui/gradio_app.py`**

Import thêm `shutil`, `time`, `from pathlib import Path`, và từ `vision_count`: `LABEL_FULL`, `LABEL_NONE`, `LABEL_NUMBER`, `get_detector`, `is_model_downloaded`. Bỏ `ObjectDetector` khỏi import.

Thêm dưới `SOURCE_WEBCAM`:

```python
# Các kiểu nhãn cho người dùng chọn: (chữ hiển thị, giá trị)
LABEL_STYLE_CHOICES = [
    ("Đầy đủ: #1 người 87%", LABEL_FULL),
    ("Chỉ số thứ tự", LABEL_NUMBER),
    ("Chỉ khung", LABEL_NONE),
]

# Thư mục chứa file tải về của mọi lần đếm; thư mục con cũ hơn 1 giờ sẽ bị xóa
EXPORT_ROOT = Path(tempfile.gettempdir()) / "vision_count"


def new_export_dir(root: Path = EXPORT_ROOT, max_age_seconds: int = 3600) -> Path:
    """Tạo thư mục mới cho lần đếm này, đồng thời xóa các thư mục cũ để không đầy ổ đĩa."""
    root.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - max_age_seconds
    for child in root.iterdir():
        if child.is_dir() and child.stat().st_mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
    return Path(tempfile.mkdtemp(dir=root))
```

Đổi `def build_app(detector: ObjectDetector)` thành `def build_app() -> gr.Blocks:`, docstring "Tạo giao diện. Model được nạp qua get_detector (mỗi model một lần)."

Đổi chữ ký hàm xử lý thành `count_objects(source_mode, image_path, webcam_image, model_key, confidence, selected_classes, label_style)`. Đặt các dòng sau ngay trước khối `try:`:

```python
        model_name = config.AVAILABLE_MODELS.get(model_key, (model_key,))[0]
        if not is_model_downloaded(model_key):
            gr.Info(f"Đang tải model {model_name} lần đầu, vui lòng chờ...")
```

Trong `try:`, nạp model trước rồi mới đọc ảnh. Đổi `except` để bắt thêm lỗi tải model:

```python
        try:
            detector = get_detector(model_key)
            image = load_image(source)
            result = detector.detect(image, confidence=confidence, classes=selected_classes or None)
        except (InvalidImageError, ValueError) as exc:
            # gr.Error hiện thông báo lỗi màu đỏ trên giao diện thay vì làm sập app
            raise gr.Error(str(exc)) from exc
        except Exception as exc:  # Ví dụ: tải model thất bại vì mất mạng
            raise gr.Error(f"Không nạp được model {model_name}: {exc}") from exc
```

Vẽ: `draw_detections(image, result.detections, label_fn=vi_label, label_style=label_style)`.

Tóm tắt có tên model: trong nhánh có vật thể, dùng `summary = f"### Tổng cộng: {result.total} vật thể ({len(result.counts)} loại)\n_Model: {model_name}_"`; nhánh không có vật thể thêm `f"\n\n_Model: {model_name}_"` vào cuối.

Thư mục xuất: `files = export_result(annotated, result, new_export_dir())`.

`gr.Blocks(title="vision-count", delete_cache=(3600, 3600))`: Gradio tự xóa bản sao file cũ hơn 1 giờ, kiểm tra mỗi giờ.

Ô lọc: `choices=[(display_label(name), name) for name in get_detector().class_names]`.

Thêm 2 ô nhập, đặt ngay trên `confidence_input`:

```python
                model_input = gr.Dropdown(
                    choices=[(label, key) for key, (label, _) in config.AVAILABLE_MODELS.items()],
                    value=config.DEFAULT_MODEL_KEY,
                    label="Model",
                    info="Small chính xác hơn với vật nhỏ/bị che. Lần đầu chọn sẽ tải file (~19MB).",
                )
```

và ngay dưới `class_input`:

```python
                label_style_input = gr.Radio(
                    choices=LABEL_STYLE_CHOICES,
                    value=LABEL_FULL,
                    label="Kiểu nhãn trên ảnh",
                    info="Khi có nhiều vật, chọn kiểu gọn để nhãn không đè lên nhau.",
                )
```

`inputs` của `run_button.click`: `[source_mode, image_input, webcam_input, model_input, confidence_input, class_input, label_style_input]`.

Sửa `app.py`:

```python
from ui.gradio_app import build_app
from vision_count import get_detector


def main():
    print("Đang nạp model YOLO (lần đầu sẽ tải file model về, mất vài giây)...")
    get_detector()  # Nạp sẵn model mặc định để lần bấm "Đếm" đầu tiên không phải chờ
    app = build_app()
```

- [ ] **Bước 4: Chạy toàn bộ test, phải PASS**: `.venv/bin/pytest -q`

- [ ] **Bước 5: Cập nhật README**

- Mục "Cách dùng": thêm bước chọn **Model** (nano/small, lần đầu chọn small sẽ tải ~19MB) và **Kiểu nhãn trên ảnh**.
- Mẹo ô lọc: gõ tên tiếng Anh ("person") cũng tìm được.
- Mục 5 (dùng trong code): ví dụ `get_detector("small")` và `label_style=LABEL_NUMBER`.
- Cây thư mục: thêm `registry.py`.
- Mục lỗi thường gặp: "Không nạp được model Small" → cần mạng ở lần đầu, hoặc tự tải `yolo11s.pt` đặt vào `models/`.

- [ ] **Bước 6: Kiểm tra trên trình duyệt**

Chạy `.venv/bin/python app.py`, tải `samples/bus.jpg`. Kỳ vọng:
- Chọn Small → bấm Đếm → tóm tắt ghi "_Model: Small: chính xác hơn, chậm hơn_".
- Chọn "Chỉ số thứ tự" → nhãn trên ảnh chỉ còn số 1, 2, 3...
- Chọn "Chỉ khung" → không còn nhãn.

- [ ] **Bước 7: Commit**: `git commit -m "Giao diện: chọn model nano/small, chọn kiểu nhãn, tự dọn thư mục tạm"`
