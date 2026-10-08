# Tên tiếng Việt và tải kết quả về: Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Mục tiêu:** Giao diện hiện tên loại vật bằng tiếng Việt (trên ảnh, bảng, ô lọc), và có nút tải về ảnh đã khoanh khung kèm file CSV số lượng và chi tiết từng khung.

**Kiến trúc:** Phần lõi `ObjectDetector` giữ nguyên tên tiếng Anh. Đây là "mã" ổn định của model, API sau này vẫn trả về tên tiếng Anh. Việc dịch nằm ở một module riêng `labels_vi.py`, được dùng ở lớp hiển thị: hàm vẽ nhận thêm tham số `label_fn`, module `export.py` ghi cả tên Việt lẫn tên gốc. Gradio chỉ nối các mảnh lại với nhau.

**Công nghệ:** Python 3.12, Pillow (vẽ chữ bằng font hệ thống), module `csv` có sẵn, Gradio 6 (`gr.File` để tải về), pytest.

**Spec:** Không có file spec riêng. Yêu cầu là mục 1 và 2 trong danh sách đề xuất đã được người dùng đồng ý trong hội thoại ngày 2026-10-08:
1. Hiện tên loại vật bằng tiếng Việt ("người", "ô tô", "chai" thay vì "person", "car", "bottle").
2. Tải kết quả về: ảnh đã khoanh khung và file CSV bảng số lượng.

## Ràng buộc chung

- Không dùng dịch vụ hay API bên ngoài; mọi thứ chạy offline trên máy.
- Không thêm thư viện mới vào `requirements.txt`.
- `vision_count/detector.py` không đổi hành vi: `Detection.label` và `DetectionResult.counts` vẫn dùng tên tiếng Anh của model.
- Chú thích code bằng tiếng Việt, theo phong cách các file hiện có.
- File CSV phải mở đúng dấu tiếng Việt trong Excel, tức là dùng mã hóa `utf-8-sig` (có BOM).

## Điểm cần chú ý khi review

1. **Font không có dấu tiếng Việt:** `ImageFont.load_default()` vẽ "người" thành "ng□□i" (đã kiểm chứng). Phải tìm font hệ thống: Arial trên macOS/Windows, DejaVu/Liberation/Noto trên Linux. Không tìm được font nào thì vẫn chạy được, không crash. Test: `test_find_font_returns_truetype_font_on_this_machine`.
2. **Model fine-tune có tên loại không có trong bảng dịch:** phải giữ nguyên tên gốc, không lỗi. Test: `test_unknown_label_falls_back_to_original`.
3. **Mở CSV bằng Excel:** thiếu BOM thì dấu tiếng Việt bị vỡ. Test: `test_csv_files_have_utf8_bom`.
4. **Xuất khi không có vật thể nào:** vẫn tạo đủ 3 file, CSV có tiêu đề và dòng tổng bằng 0. Test: `test_export_empty_result`.
5. **Đếm nhiều lần liên tiếp:** file lần sau không được ghi đè file lần trước đang tải dở. Mỗi lần xuất dùng thư mục tạm riêng; tên file có dấu thời gian. Test: `test_export_file_names_have_timestamp`.

---

## Cấu trúc file

| File | Việc | Trách nhiệm |
|---|---|---|
| `vision_count/labels_vi.py` | Tạo mới | Bảng dịch 80 tên COCO, hàm `vi_label()` và `display_label()` |
| `vision_count/drawing.py` | Sửa | Thêm tham số `label_fn`, tìm font hỗ trợ tiếng Việt |
| `vision_count/export.py` | Tạo mới | Ghi ảnh + 2 file CSV vào một thư mục |
| `vision_count/__init__.py` | Sửa | Xuất thêm `vi_label`, `display_label`, `export_result` |
| `ui/gradio_app.py` | Sửa | Hiển thị tiếng Việt, thêm ô "Tải kết quả về" |
| `tests/test_labels_vi.py` | Tạo mới | Test bảng dịch |
| `tests/test_drawing.py` | Tạo mới | Test font và `label_fn` |
| `tests/test_export.py` | Tạo mới | Test xuất file |
| `README.md` | Sửa | Mô tả tính năng mới |

---

### Task 1: Bảng dịch tên tiếng Việt

**Files:**
- Tạo: `vision_count/labels_vi.py`
- Sửa: `vision_count/__init__.py`
- Test: `tests/test_labels_vi.py`

**Interfaces:**
- Produces: `vi_label(label: str) -> str` trả tên tiếng Việt, hoặc chính `label` nếu không có trong bảng. `display_label(label: str) -> str` trả `"người (person)"`, hoặc `label` nếu không có bản dịch. `COCO_VI: dict[str, str]`.

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
"""Test bảng dịch tên loại vật sang tiếng Việt."""

from vision_count import ObjectDetector, display_label, vi_label
from vision_count.labels_vi import COCO_VI


def test_translates_common_labels():
    assert vi_label("person") == "người"
    assert vi_label("car") == "ô tô"
    assert vi_label("bottle") == "chai"


def test_unknown_label_falls_back_to_original():
    # Model fine-tune có thể có loại mới chưa có trong bảng dịch
    assert vi_label("oc_vit") == "oc_vit"
    assert display_label("oc_vit") == "oc_vit"


def test_display_label_shows_both_names():
    assert display_label("person") == "người (person)"


def test_covers_all_80_coco_classes():
    names = ObjectDetector().class_names
    missing = [n for n in names if n not in COCO_VI]
    assert missing == []
    assert len(COCO_VI) == 80
```

- [ ] **Bước 2: Chạy test, phải FAIL**

Chạy: `.venv/bin/pytest tests/test_labels_vi.py -v`
Kỳ vọng: lỗi `ImportError: cannot import name 'display_label'`.

- [ ] **Bước 3: Viết `vision_count/labels_vi.py`**

```python
"""Bảng dịch tên 80 loại vật COCO sang tiếng Việt.

Model vẫn dùng tên tiếng Anh làm "mã" (ổn định cho API). Bảng này chỉ dùng
khi hiển thị cho người dùng.
"""

COCO_VI = {
    "person": "người", "bicycle": "xe đạp", "car": "ô tô", "motorcycle": "xe máy",
    "airplane": "máy bay", "bus": "xe buýt", "train": "tàu hỏa", "truck": "xe tải",
    "boat": "thuyền", "traffic light": "đèn giao thông", "fire hydrant": "trụ cứu hỏa",
    "stop sign": "biển dừng", "parking meter": "đồng hồ đỗ xe", "bench": "ghế băng",
    "bird": "chim", "cat": "mèo", "dog": "chó", "horse": "ngựa", "sheep": "cừu",
    "cow": "bò", "elephant": "voi", "bear": "gấu", "zebra": "ngựa vằn",
    "giraffe": "hươu cao cổ", "backpack": "ba lô", "umbrella": "ô (dù)",
    "handbag": "túi xách", "tie": "cà vạt", "suitcase": "vali", "frisbee": "đĩa ném",
    "skis": "ván trượt tuyết", "snowboard": "ván trượt tuyết đơn",
    "sports ball": "quả bóng", "kite": "diều", "baseball bat": "gậy bóng chày",
    "baseball glove": "găng bóng chày", "skateboard": "ván trượt",
    "surfboard": "ván lướt sóng", "tennis racket": "vợt tennis", "bottle": "chai",
    "wine glass": "ly rượu", "cup": "cốc", "fork": "nĩa", "knife": "dao",
    "spoon": "thìa", "bowl": "bát", "banana": "chuối", "apple": "táo",
    "sandwich": "bánh mì kẹp", "orange": "cam", "broccoli": "súp lơ xanh",
    "carrot": "cà rốt", "hot dog": "xúc xích kẹp", "pizza": "pizza",
    "donut": "bánh donut", "cake": "bánh ngọt", "chair": "ghế", "couch": "ghế sofa",
    "potted plant": "chậu cây", "bed": "giường", "dining table": "bàn ăn",
    "toilet": "bồn cầu", "tv": "tivi", "laptop": "laptop", "mouse": "chuột máy tính",
    "remote": "điều khiển từ xa", "keyboard": "bàn phím", "cell phone": "điện thoại",
    "microwave": "lò vi sóng", "oven": "lò nướng", "toaster": "máy nướng bánh mì",
    "sink": "bồn rửa", "refrigerator": "tủ lạnh", "book": "sách", "clock": "đồng hồ",
    "vase": "bình hoa", "scissors": "kéo", "teddy bear": "gấu bông",
    "hair drier": "máy sấy tóc", "toothbrush": "bàn chải đánh răng",
}


def vi_label(label: str) -> str:
    """Trả về tên tiếng Việt; nếu chưa có bản dịch thì giữ nguyên tên gốc."""
    return COCO_VI.get(label, label)


def display_label(label: str) -> str:
    """Tên hiển thị kèm tên gốc, ví dụ 'người (person)'. Dùng trong ô lọc và bảng."""
    vi = COCO_VI.get(label)
    return f"{vi} ({label})" if vi else label
```

Trong `vision_count/__init__.py`, thêm import `from vision_count.labels_vi import display_label, vi_label` và thêm `"display_label"`, `"vi_label"` vào `__all__`.

- [ ] **Bước 4: Chạy test, phải PASS**

Chạy: `.venv/bin/pytest tests/test_labels_vi.py -v`
Kỳ vọng: 4 passed.

- [ ] **Bước 5: Commit**

```bash
git add vision_count/labels_vi.py vision_count/__init__.py tests/test_labels_vi.py
git commit -m "Thêm bảng dịch tên 80 loại vật COCO sang tiếng Việt"
```

---

### Task 2: Vẽ nhãn tiếng Việt lên ảnh

**Files:**
- Sửa: `vision_count/drawing.py`
- Test: `tests/test_drawing.py`

**Interfaces:**
- Consumes: `vi_label` (Task 1) chỉ dùng trong test.
- Produces: `draw_detections(image, detections, label_fn: Callable[[str], str] | None = None) -> Image.Image`. `label_fn=None` vẽ tên gốc như cũ. `find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont`.

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
"""Test phần vẽ khung và nhãn."""

from PIL import Image, ImageFont

from vision_count import Detection, draw_detections, vi_label
from vision_count.drawing import find_font


def _sample_detection():
    return Detection(label="person", class_id=0, confidence=0.9, box=(10, 30, 120, 180))


def test_find_font_returns_truetype_font_on_this_machine():
    # Máy dev (macOS) có Arial hỗ trợ tiếng Việt; font mặc định của Pillow thì không
    assert isinstance(find_font(20), ImageFont.FreeTypeFont)


def test_label_fn_is_used_for_text():
    seen = []

    def spy(label):
        seen.append(label)
        return vi_label(label)

    draw_detections(Image.new("RGB", (200, 200)), [_sample_detection()], label_fn=spy)
    assert seen == ["person"]


def test_default_draws_without_label_fn():
    out = draw_detections(Image.new("RGB", (200, 200)), [_sample_detection()])
    assert out.size == (200, 200)
```

- [ ] **Bước 2: Chạy test, phải FAIL**

Chạy: `.venv/bin/pytest tests/test_drawing.py -v`
Kỳ vọng: `ImportError: cannot import name 'find_font'`.

- [ ] **Bước 3: Sửa `vision_count/drawing.py`**

Thêm danh sách font và hàm `find_font`, ngay dưới `PALETTE`:

```python
# Font có đủ dấu tiếng Việt. Font mặc định của Pillow KHÔNG có, sẽ vẽ dấu thành ô vuông.
# Pillow tự tìm các tên này trong thư mục font của hệ điều hành.
VIETNAMESE_FONTS = [
    "Arial.ttf",  # macOS, Windows
    "arial.ttf",  # Windows (tên chữ thường)
    "DejaVuSans.ttf",  # Linux
    "LiberationSans-Regular.ttf",  # Linux
    "NotoSans-Regular.ttf",  # Linux
]


@lru_cache(maxsize=16)
def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Tìm font hỗ trợ tiếng Việt; không có thì dùng font mặc định (mất dấu nhưng không lỗi)."""
    for name in VIETNAMESE_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)
```

Thêm import `from functools import lru_cache` và `from typing import Callable`. Đổi chữ ký hàm và hai dòng dùng font/nhãn:

```python
def draw_detections(
    image: Image.Image,
    detections: list[Detection],
    label_fn: Callable[[str], str] | None = None,
) -> Image.Image:
    """Trả về bản sao của ảnh, đã vẽ khung + số thứ tự + tên + độ tin cậy.

    label_fn: hàm đổi tên loại vật trước khi vẽ (vd vi_label để hiện tiếng Việt).
    """
    ...
    font = find_font(max(12, round(18 * scale)))
    ...
        name = label_fn(det.label) if label_fn else det.label
        text = f"#{index} {name} {det.confidence:.0%}"
```

- [ ] **Bước 4: Chạy toàn bộ test, phải PASS**

Chạy: `.venv/bin/pytest -v`
Kỳ vọng: tất cả passed, kể cả `test_draw_detections_keeps_size_and_original` cũ.

- [ ] **Bước 5: Commit**

```bash
git add vision_count/drawing.py tests/test_drawing.py
git commit -m "Vẽ nhãn tiếng Việt: thêm label_fn và tự tìm font có dấu"
```

---

### Task 3: Xuất kết quả ra file

**Files:**
- Tạo: `vision_count/export.py`
- Sửa: `vision_count/__init__.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `DetectionResult`, `Detection` (detector.py), `vi_label` (Task 1).
- Produces: `export_result(annotated: Image.Image, result: DetectionResult, out_dir: str | Path, timestamp: datetime | None = None) -> list[Path]` trả về `[ảnh .jpg, so_luong .csv, chi_tiet .csv]` theo đúng thứ tự đó.

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
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
```

- [ ] **Bước 2: Chạy test, phải FAIL**

Chạy: `.venv/bin/pytest tests/test_export.py -v`
Kỳ vọng: `ImportError: cannot import name 'export_result'`.

- [ ] **Bước 3: Viết `vision_count/export.py`**

```python
"""Xuất kết quả đếm ra file: ảnh đã khoanh khung + 2 file CSV.

CSV ghi bằng 'utf-8-sig' (có BOM) để Excel mở không bị lỗi dấu tiếng Việt.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from PIL import Image

from vision_count.detector import DetectionResult
from vision_count.labels_vi import vi_label

COUNT_HEADER = ["Loại vật thể", "Tên gốc (model)", "Số lượng"]
DETAIL_HEADER = ["#", "Loại vật thể", "Tên gốc (model)", "Độ tin cậy", "x1", "y1", "x2", "y2"]


def export_result(
    annotated: Image.Image,
    result: DetectionResult,
    out_dir: str | Path,
    timestamp: datetime | None = None,
) -> list[Path]:
    """Ghi 3 file vào out_dir và trả về danh sách đường dẫn [ảnh, số lượng, chi tiết]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Dấu thời gian trong tên file để các lần tải về không trùng tên nhau
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")

    image_path = out_dir / f"ket_qua_{stamp}.jpg"
    annotated.convert("RGB").save(image_path, quality=95)

    counts_path = out_dir / f"so_luong_{stamp}.csv"
    count_rows = [[vi_label(label), label, n] for label, n in result.counts.items()]
    count_rows.append(["Tổng cộng", "", result.total])
    _write_csv(counts_path, COUNT_HEADER, count_rows)

    detail_path = out_dir / f"chi_tiet_{stamp}.csv"
    detail_rows = [
        [i, vi_label(d.label), d.label, d.confidence, *d.box]
        for i, d in enumerate(result.detections, start=1)
    ]
    _write_csv(detail_path, DETAIL_HEADER, detail_rows)

    return [image_path, counts_path, detail_path]


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
```

Trong `vision_count/__init__.py`, thêm `from vision_count.export import export_result` và `"export_result"` vào `__all__`.

- [ ] **Bước 4: Chạy test, phải PASS**

Chạy: `.venv/bin/pytest -v`
Kỳ vọng: tất cả passed.

- [ ] **Bước 5: Commit**

```bash
git add vision_count/export.py vision_count/__init__.py tests/test_export.py
git commit -m "Thêm export_result: xuất ảnh kết quả và CSV số lượng, chi tiết"
```

---

### Task 4: Nối vào giao diện Gradio và cập nhật README

**Files:**
- Sửa: `ui/gradio_app.py`
- Sửa: `README.md`

**Interfaces:**
- Consumes: `vi_label`, `display_label` (Task 1), `draw_detections(..., label_fn=)` (Task 2), `export_result` (Task 3).

- [ ] **Bước 1: Sửa `ui/gradio_app.py`**

Import thêm:

```python
import tempfile

from vision_count import (
    InvalidImageError, ObjectDetector, config, display_label, draw_detections,
    export_result, load_image, vi_label,
)
```

Trong `count_objects`:

```python
        annotated = draw_detections(image, result.detections, label_fn=vi_label)
        ...
        counts_table = pd.DataFrame(
            [[display_label(label), n] for label, n in result.counts.items()], columns=COUNT_COLUMNS
        )
        detail_table = pd.DataFrame(
            [
                [i, display_label(d.label), f"{d.confidence:.0%}", ", ".join(f"{v:.0f}" for v in d.box)]
                for i, d in enumerate(result.detections, start=1)
            ],
            columns=DETAIL_COLUMNS,
        )
        # Mỗi lần đếm ghi vào một thư mục tạm riêng, để không ghi đè file của lần trước
        files = export_result(annotated, result, tempfile.mkdtemp(prefix="vision_count_"))
        return annotated, summary, counts_table, detail_table, [str(p) for p in files]
```

Ô lọc: `choices=[(display_label(name), name) for name in detector.class_names]`. Gradio hiện chữ thứ nhất, nhưng giá trị gửi về là tên gốc, nên `detector.detect(classes=...)` không phải đổi gì.

Dưới Accordion, thêm `download_output = gr.File(label="Tải kết quả về (ảnh + CSV)", file_count="multiple", interactive=False)` và thêm `download_output` vào `outputs` của `run_button.click`.

- [ ] **Bước 2: Cập nhật README**

Trong mục "Cách dùng → Kết quả gồm", thêm ý tải về 3 file (ảnh JPG, `so_luong_*.csv`, `chi_tiet_*.csv`; mở được bằng Excel). Ghi chú tên hiển thị dạng "người (person)". Trong mục 5 (dùng trong code), thêm ví dụ `vi_label` và `export_result`. Cập nhật cây thư mục với `labels_vi.py` và `export.py`.

- [ ] **Bước 3: Kiểm tra thủ công trên trình duyệt**

Chạy `.venv/bin/python app.py`, mở http://127.0.0.1:7860, tải `samples/bus.jpg` lên rồi bấm **Đếm**. Kỳ vọng:
- Ảnh có nhãn "#1 xe buýt 94%", dấu tiếng Việt hiển thị đúng.
- Bảng có dòng "người (person) | 4" và "xe buýt (bus) | 1".
- Ô tải về có 3 file, tải được.
- Ô lọc hiện "người (person)"; chọn mục này thì chỉ đếm người.

- [ ] **Bước 4: Chạy toàn bộ test, rồi commit**

```bash
.venv/bin/pytest -q
git add ui/gradio_app.py README.md
git commit -m "Giao diện: hiện tên tiếng Việt và thêm nút tải kết quả về"
```
