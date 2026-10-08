# Giai đoạn 2: Vùng đếm, vật nhỏ, nhiều ảnh, lịch sử: Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Mục tiêu:** Thêm 4 chức năng giúp đếm thực tế tốt hơn:
1. **Vùng đếm:** bấm 2 điểm trên ảnh kết quả để khoanh một hình chữ nhật, chỉ đếm vật nằm trong đó.
2. **Chế độ vật nhỏ:** chia ảnh thành nhiều ô rồi nhận diện từng ô, kiểu SAHI tự viết.
3. **Đếm nhiều ảnh:** tải nhiều ảnh cùng lúc, nhận bảng tổng hợp và file CSV.
4. **Lịch sử đếm:** lưu mỗi lần đếm vào SQLite trên máy và xem lại được.

**Kiến trúc:** Mỗi chức năng là một module lõi riêng trong `vision_count/`, không phụ thuộc Gradio:
- `region.py`: vùng đếm.
- `tiling.py`: chia ô và gộp khung trùng.
- `batch.py`: đếm nhiều ảnh.
- `history.py`: lưu lịch sử.

`ObjectDetector.detect()` thêm một tham số tùy chọn `tiled=False`, nên mặc định vẫn chạy như cũ. Về giao diện, các hàm xử lý được tách ra `ui/handlers.py` (hàm thường, test trực tiếp được). `ui/gradio_app.py` chỉ còn bố cục và nối sự kiện. Phần cài đặt dùng chung đưa lên đầu trang, bên dưới là 3 tab: "Một ảnh", "Nhiều ảnh", "Lịch sử".

**Công nghệ:** Python 3.12, Ultralytics, Pillow, `sqlite3` và `csv` có sẵn trong Python, Gradio 6 (`gr.Image.select`, `gr.Gallery`), pytest.

**Spec:** Không có file spec riêng. Yêu cầu là nhóm 2 (mục 4-7) trong danh sách đề xuất ngày 2026-10-08. Người dùng chọn: vùng đếm theo cách bấm 2 điểm thành hình chữ nhật; SAHI tự viết, không thêm thư viện.

## Ràng buộc chung

- Chạy offline. Không thêm thư viện vào `requirements.txt`.
- `detect(image, confidence, classes)` gọi như cũ vẫn cho đúng kết quả cũ.
- Lịch sử chỉ lưu số liệu (thời gian, nguồn, model, chế độ, số lượng), **không lưu ảnh**, tránh tốn ổ đĩa và giữ riêng tư.
- File SQLite nằm ở `data/history.db` và được thêm vào `.gitignore`.
- Chú thích code bằng tiếng Việt.

## Quy ước

- **Vật thuộc vùng đếm** khi **tâm** khung nằm trong vùng (tính cả cạnh). Vật nằm vắt qua cạnh vùng được tính theo tâm.
- **Gộp khung trùng khi chia ô:** hai khung cùng loại bị coi là một nếu `phần giao / diện tích khung nhỏ hơn > 0.6` (IoS). Khi đó giữ khung có độ tin cậy cao hơn. Nhờ vậy, nửa người bị cắt ở mép ô sẽ được gộp vào khung nguyên người ở ô bên cạnh.
- **Chế độ vật nhỏ** = nhận diện cả ảnh (bắt vật to) + từng ô 640×640 chồng nhau 20% (bắt vật nhỏ), rồi gộp khung trùng. Ảnh nhỏ hơn hoặc bằng 640 thì chạy như thường.

## Điểm cần chú ý khi review

1. **Bấm 2 lần vào cùng một điểm, hoặc kéo ra vùng quá mỏng:** phải từ chối kèm thông báo, không tạo vùng rỗng. Test: `test_add_region_point_rejects_tiny_region`.
2. **Đổi sang ảnh khác kích thước sau khi đã chọn vùng:** vùng vẫn giữ theo tọa độ pixel. Nếu nằm ngoài ảnh thì kết quả là 0, và tóm tắt phải nói rõ đang đếm trong vùng để người dùng hiểu vì sao. Test: `test_count_single_with_region_mentions_region`.
3. **Nhiều ảnh, trong đó có file hỏng:** file hỏng ghi lỗi vào bảng, các ảnh khác vẫn được đếm. Test: `test_count_many_keeps_going_after_bad_file`.
4. **Gộp trùng xóa nhầm hai vật khác loại nằm chồng nhau:** khác loại thì không gộp. Test: `test_merge_overlaps_keeps_different_classes`.
5. **Bấm vào ảnh kết quả khi chưa đếm lần nào:** phải có hướng dẫn, không lỗi. Test: `test_add_region_point_without_image`.

---

## Cấu trúc file

| File | Việc | Trách nhiệm |
|---|---|---|
| `vision_count/region.py` | Tạo | `normalize_region`, `filter_by_region` |
| `vision_count/drawing.py` | Sửa | `draw_detections(..., region=)`, `draw_region()` |
| `vision_count/tiling.py` | Tạo | `make_tiles`, `merge_overlaps` |
| `vision_count/detector.py` | Sửa | `detect(..., tiled=False)`, tách `_predict` |
| `vision_count/batch.py` | Tạo | `BatchItem`, `count_many`, `write_batch_csv` |
| `vision_count/history.py` | Tạo | `HistoryEntry`, `HistoryStore` |
| `vision_count/config.py` | Sửa | `TILE_OVERLAP`, `HISTORY_DB` |
| `vision_count/export.py` | Sửa | Đổi `_write_csv` thành `write_csv` (dùng lại cho batch) |
| `vision_count/__init__.py` | Sửa | Xuất tên mới |
| `ui/handlers.py` | Tạo | Các hàm xử lý sự kiện, test được trực tiếp |
| `ui/gradio_app.py` | Sửa | Bố cục 3 tab + nối sự kiện |
| `tests/test_region.py`, `test_tiling.py`, `test_batch.py`, `test_history.py`, `test_handlers.py` | Tạo | |
| `tests/test_ui.py` | Sửa | Chỉ còn test dựng app |
| `.gitignore`, `README.md` | Sửa | |

---

### Task 1: Vùng đếm (lõi + vẽ)

**Files:** Tạo `vision_count/region.py`. Sửa `vision_count/drawing.py`, `vision_count/__init__.py`. Test `tests/test_region.py`.

**Interfaces:**
- Produces:
  - `Region = tuple[float, float, float, float]`.
  - `normalize_region(p1: tuple[float, float], p2: tuple[float, float]) -> Region`.
  - `filter_by_region(result: DetectionResult, region: Region | None) -> DetectionResult`.
  - `draw_region(image, region: Region | None = None, point: tuple[float, float] | None = None) -> Image.Image`.
  - `draw_detections(..., region: Region | None = None)`.
  - `REGION_COLOR = (255, 214, 0)`.

- [ ] **Bước 1: Viết test `tests/test_region.py` (sẽ fail)**

```python
"""Test vùng đếm."""

from PIL import Image, ImageChops

from vision_count import (
    Detection,
    DetectionResult,
    draw_detections,
    draw_region,
    filter_by_region,
    normalize_region,
)
from vision_count.drawing import REGION_COLOR


def _result():
    dets = [
        Detection("person", 0, 0.9, (0.0, 0.0, 20.0, 20.0)),  # tâm (10, 10): trong vùng
        Detection("person", 0, 0.8, (80.0, 80.0, 100.0, 100.0)),  # tâm (90, 90): ngoài vùng
        Detection("car", 2, 0.7, (30.0, 30.0, 70.0, 50.0)),  # tâm (50, 40): trong vùng
    ]
    return DetectionResult(detections=dets, counts={"person": 2, "car": 1}, image_width=100, image_height=100)


def test_normalize_region_orders_corners():
    assert normalize_region((60, 70), (5, 10)) == (5, 10, 60, 70)


def test_filter_keeps_only_centers_inside_and_recounts():
    filtered = filter_by_region(_result(), (0, 0, 60, 60))
    assert [d.confidence for d in filtered.detections] == [0.9, 0.7]
    assert filtered.counts == {"person": 1, "car": 1}
    assert filtered.total == 2
    assert (filtered.image_width, filtered.image_height) == (100, 100)


def test_filter_center_on_edge_counts():
    filtered = filter_by_region(_result(), (10, 10, 50, 40))  # đúng bằng tâm 2 khung
    assert filtered.total == 2


def test_filter_without_region_returns_same():
    result = _result()
    assert filter_by_region(result, None) is result


def test_draw_detections_shows_region():
    img = Image.new("RGB", (100, 100))
    with_region = draw_detections(img, [], region=(10, 10, 60, 60))
    assert with_region.getpixel((10, 30)) == REGION_COLOR
    assert ImageChops.difference(with_region, img).getbbox() is not None


def test_draw_region_marks_point_and_keeps_original():
    img = Image.new("RGB", (100, 100))
    marked = draw_region(img, point=(50, 50))
    assert marked.getpixel((50, 50)) == REGION_COLOR
    assert img.getpixel((50, 50)) == (0, 0, 0)
```

- [ ] **Bước 2:** Chạy `.venv/bin/pytest tests/test_region.py -q`. Kỳ vọng: FAIL với `ImportError: cannot import name 'draw_region'`.

- [ ] **Bước 3: Viết code**

`vision_count/region.py`:

```python
"""Vùng đếm: chỉ đếm vật có TÂM khung nằm trong một hình chữ nhật (tính cả cạnh)."""

from __future__ import annotations

from collections import Counter

from vision_count.detector import DetectionResult

# (x1, y1, x2, y2) theo pixel của ảnh gốc, x1 <= x2 và y1 <= y2
Region = tuple[float, float, float, float]


def normalize_region(p1: tuple[float, float], p2: tuple[float, float]) -> Region:
    """Tạo vùng từ 2 điểm bất kỳ (người dùng có thể bấm góc nào trước cũng được)."""
    (ax, ay), (bx, by) = p1, p2
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


def filter_by_region(result: DetectionResult, region: Region | None) -> DetectionResult:
    """Giữ lại các vật có tâm nằm trong vùng và đếm lại. region=None thì trả về nguyên kết quả."""
    if region is None:
        return result
    x1, y1, x2, y2 = region
    kept = [
        d
        for d in result.detections
        if x1 <= (d.box[0] + d.box[2]) / 2 <= x2 and y1 <= (d.box[1] + d.box[3]) / 2 <= y2
    ]
    return DetectionResult(
        detections=kept,
        counts=dict(Counter(d.label for d in kept).most_common()),
        image_width=result.image_width,
        image_height=result.image_height,
    )
```

Trong `vision_count/drawing.py`, thêm sau `PALETTE`:

```python
# Màu vàng cho vùng đếm, khác hẳn màu các khung vật thể
REGION_COLOR = (255, 214, 0)
```

Thêm hàm `draw_region` (đặt trước `draw_detections`) và hàm phụ `_line_width`:

```python
def _line_width(image: Image.Image) -> int:
    """Độ dày nét tỉ lệ theo kích thước ảnh, để ảnh to hay nhỏ đều dễ nhìn."""
    return max(2, round(3 * max(image.size) / 1000))


def draw_region(
    image: Image.Image,
    region: tuple[float, float, float, float] | None = None,
    point: tuple[float, float] | None = None,
) -> Image.Image:
    """Vẽ vùng đếm (hình chữ nhật vàng) và/hoặc một điểm đánh dấu lên BẢN SAO của ảnh."""
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    width = _line_width(marked)
    if region is not None:
        draw.rectangle(region, outline=REGION_COLOR, width=width)
    if point is not None:
        r = width * 3
        x, y = point
        draw.ellipse((x - r, y - r, x + r, y + r), fill=REGION_COLOR)
    return marked
```

Trong `draw_detections`, thêm tham số `region: tuple[float, float, float, float] | None = None` (ghi vào docstring: "region: vùng đếm, vẽ khung vàng"). Thay 2 dòng `annotated = image.convert("RGB").copy()` / `draw = ImageDraw.Draw(annotated)` bằng:

```python
    annotated = draw_region(image, region)  # Bản sao của ảnh (kèm vùng đếm nếu có)
    draw = ImageDraw.Draw(annotated)
```

và đổi `line_width = max(2, round(3 * scale))` thành `line_width = _line_width(annotated)`.

Trong `vision_count/__init__.py`, xuất thêm `draw_region` (từ drawing), `Region`, `filter_by_region`, `normalize_region` (từ region).

- [ ] **Bước 4:** Chạy `.venv/bin/pytest -q`. Kỳ vọng: tất cả PASS (kể cả `test_none_style_draws_only_box` cũ, vì độ dày nét không đổi).
- [ ] **Bước 5:** Commit `Thêm vùng đếm: lọc theo tâm khung và vẽ vùng`.

---

### Task 2: Chế độ vật nhỏ (chia ô)

**Files:** Tạo `vision_count/tiling.py`. Sửa `vision_count/detector.py`, `vision_count/config.py`. Test `tests/test_tiling.py`.

**Interfaces:**
- Produces:
  - `make_tiles(width: int, height: int, tile: int = 640, overlap: float = 0.2) -> list[tuple[int, int, int, int]]`.
  - `merge_overlaps(detections: list[Detection], ios_threshold: float = 0.6) -> list[Detection]`.
  - `ObjectDetector.detect(image, confidence=..., classes=None, tiled: bool = False)`.
  - `config.TILE_OVERLAP = 0.2`.

- [ ] **Bước 1: Viết test `tests/test_tiling.py` (sẽ fail)**

```python
"""Test chế độ vật nhỏ: chia ô và gộp khung trùng."""

import pytest
from PIL import Image
from ultralytics.utils import ASSETS

from vision_count import Detection, get_detector
from vision_count.tiling import make_tiles, merge_overlaps


def test_small_image_is_one_tile():
    assert make_tiles(500, 400) == [(0, 0, 500, 400)]


def test_tiles_cover_whole_image_without_exceeding_size():
    tiles = make_tiles(2000, 1300, tile=640, overlap=0.2)
    assert all(x2 - x1 <= 640 and y2 - y1 <= 640 for x1, y1, x2, y2 in tiles)
    assert max(x2 for _, _, x2, _ in tiles) == 2000
    assert max(y2 for _, _, _, y2 in tiles) == 1300
    covered = set()
    for x1, y1, x2, y2 in tiles:
        covered.update((x, y) for x in range(x1, x2, 50) for y in range(y1, y2, 50))
    assert covered == {(x, y) for x in range(0, 2000, 50) for y in range(0, 1300, 50)}


def test_merge_overlaps_merges_partial_box_at_tile_edge():
    full = Detection("person", 0, 0.9, (100.0, 100.0, 140.0, 200.0))
    half = Detection("person", 0, 0.6, (100.0, 100.0, 140.0, 150.0))  # nửa trên, nằm trong khung đầy đủ
    assert merge_overlaps([half, full]) == [full]


def test_merge_overlaps_keeps_different_classes():
    person = Detection("person", 0, 0.9, (0.0, 0.0, 50.0, 100.0))
    bag = Detection("handbag", 26, 0.8, (10.0, 40.0, 30.0, 60.0))  # nằm trong người nhưng khác loại
    assert merge_overlaps([person, bag]) == [person, bag]


def test_merge_overlaps_keeps_separate_boxes():
    a = Detection("person", 0, 0.9, (0.0, 0.0, 50.0, 100.0))
    b = Detection("person", 0, 0.8, (60.0, 0.0, 110.0, 100.0))
    assert merge_overlaps([b, a]) == [a, b]


@pytest.fixture(scope="module")
def tiny_people_canvas():
    """Ảnh 2400x2400 có 16 người nhỏ (cao 80px) cắt từ ảnh xe buýt mẫu."""
    detector = get_detector()
    bus = Image.open(ASSETS / "bus.jpg").convert("RGB")
    people = detector.detect(bus, classes=["person"]).detections[:4]
    crops = [bus.crop(tuple(int(v) for v in d.box)) for d in people]
    canvas = Image.new("RGB", (2400, 2400), (128, 128, 128))
    for i in range(4):
        for j in range(4):
            crop = crops[(i * 4 + j) % len(crops)]
            small = crop.resize((max(1, int(crop.width * 80 / crop.height)), 80))
            canvas.paste(small, (150 + j * 560, 150 + i * 560))
    return canvas


def test_tiled_finds_small_people_normal_misses(tiny_people_canvas):
    detector = get_detector()
    normal = detector.detect(tiny_people_canvas, classes=["person"])
    tiled = detector.detect(tiny_people_canvas, classes=["person"], tiled=True)
    assert normal.total <= 2  # thu nhỏ cả ảnh về 640px thì người chỉ còn ~20px, gần như mất hết
    assert 12 <= tiled.total <= 16  # chia ô bắt được hầu hết, không đếm trùng quá số đã dán


def test_tiled_on_small_image_equals_normal():
    detector = get_detector()
    img = Image.open(ASSETS / "bus.jpg").convert("RGB").resize((600, 800))
    small = img.crop((0, 0, 600, 640))
    assert detector.detect(small, tiled=True).counts == detector.detect(small).counts
```

- [ ] **Bước 2:** Chạy `.venv/bin/pytest tests/test_tiling.py -q`. Kỳ vọng: FAIL với `ModuleNotFoundError: No module named 'vision_count.tiling'`.

- [ ] **Bước 3: Viết code**

Thêm vào cuối `vision_count/config.py`:

```python
# Chế độ vật nhỏ: các ô cạnh nhau chồng lên nhau 20% để vật nằm ở mép ô không bị bỏ sót
TILE_OVERLAP = 0.2
```

`vision_count/tiling.py`:

```python
"""Chế độ vật nhỏ (ý tưởng của SAHI): chia ảnh lớn thành các ô nhỏ chồng nhau.

YOLO luôn thu ảnh về 640px trước khi nhận diện, nên trong ảnh lớn, vật nhỏ bị thu còn vài pixel
và mất. Nhận diện từng ô 640px giữ nguyên độ phân giải nên bắt được vật nhỏ.
"""

from __future__ import annotations

from vision_count.detector import Detection


def make_tiles(width: int, height: int, tile: int = 640, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """Trả về danh sách ô (x1, y1, x2, y2) phủ kín ảnh; ô cuối mỗi hàng/cột sát mép ảnh."""
    if width <= tile and height <= tile:
        return [(0, 0, width, height)]
    step = max(1, int(tile * (1 - overlap)))

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        return sorted(set(range(0, size - tile, step)) | {size - tile})

    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in starts(height) for x in starts(width)]


def _ios(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Phần giao chia cho diện tích khung NHỎ hơn (khác IoU: bắt được nửa khung nằm trong khung đầy đủ)."""
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter_w * inter_h / smaller if smaller > 0 else 0.0


def merge_overlaps(detections: list[Detection], ios_threshold: float = 0.6) -> list[Detection]:
    """Gộp khung trùng CÙNG LOẠI giữa các ô: giữ khung tin cậy cao hơn, bỏ khung chồng lên nó quá ngưỡng."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(k.class_id != det.class_id or _ios(k.box, det.box) <= ios_threshold for k in kept):
            kept.append(det)
    return kept
```

Trong `vision_count/detector.py`:
- Thêm tham số `tiled: bool = False` vào `detect` (docstring: "tiled: chế độ vật nhỏ, chia ảnh thành ô 640px (chậm hơn)").
- Tách phần gọi `self.model.predict` và vòng lặp tạo `Detection` thành `_predict`.
- `detect` gọi `_predict` cho ảnh, hoặc cho ảnh + từng ô.

`tiling.py` import `Detection` từ `detector.py`, nên `detect` phải import `make_tiles` và `merge_overlaps` **bên trong hàm** để tránh import vòng.

```python
    def detect(self, image, confidence=config.DEFAULT_CONFIDENCE, classes=None, tiled=False) -> DetectionResult:
        ...  # (kiểm tra confidence, load_image, class_ids như cũ)
        if tiled:
            from vision_count.tiling import make_tiles, merge_overlaps  # import ở đây để tránh import vòng

            tiles = make_tiles(pil_image.width, pil_image.height, config.DEFAULT_IMAGE_SIZE, config.TILE_OVERLAP)
            # Nhận diện cả ảnh (bắt vật to) rồi từng ô (bắt vật nhỏ), cuối cùng gộp khung trùng
            detections = self._predict(pil_image, confidence, class_ids)
            if len(tiles) > 1:
                for x1, y1, x2, y2 in tiles:
                    detections += self._predict(pil_image.crop((x1, y1, x2, y2)), confidence, class_ids, x1, y1)
                detections = merge_overlaps(detections)
        else:
            detections = self._predict(pil_image, confidence, class_ids)

        # Sắp xếp khung theo độ tin cậy giảm dần
        detections.sort(key=lambda d: d.confidence, reverse=True)
        ...  # (counts và return như cũ)

    def _predict(self, image: Image.Image, confidence: float, class_ids, offset_x: float = 0, offset_y: float = 0) -> list[Detection]:
        """Chạy model trên một ảnh; cộng offset để đổi tọa độ trong ô về tọa độ ảnh gốc."""
        results = self.model.predict(
            source=image,
            conf=confidence,
            classes=class_ids,
            imgsz=config.DEFAULT_IMAGE_SIZE,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes  # Chỉ có 1 ảnh nên lấy phần tử đầu tiên
        detections = []
        # .cpu().numpy() chuyển tensor PyTorch sang mảng numpy để dễ xử lý
        for xyxy, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            class_id = int(cls)
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            detections.append(
                Detection(
                    label=self.model.names[class_id],
                    class_id=class_id,
                    confidence=round(float(conf), 4),
                    box=(round(x1 + offset_x, 1), round(y1 + offset_y, 1), round(x2 + offset_x, 1), round(y2 + offset_y, 1)),
                )
            )
        return detections
```

- [ ] **Bước 4:** Chạy `.venv/bin/pytest -q`. Kỳ vọng: tất cả PASS, kể cả các test cũ của `test_detector.py`.
- [ ] **Bước 5:** Commit `Thêm chế độ vật nhỏ: chia ô 640px và gộp khung trùng`.

---

### Task 3: Đếm nhiều ảnh (lõi)

**Files:** Tạo `vision_count/batch.py`. Sửa `vision_count/export.py` (đổi `_write_csv` thành `write_csv`), `vision_count/__init__.py`. Test `tests/test_batch.py`.

**Interfaces:**
- Consumes: `ObjectDetector.detect(..., tiled=)` (Task 2), `vi_label`, `write_csv`.
- Produces:
  - `BatchItem(name: str, result: DetectionResult | None = None, error: str | None = None)`.
  - `count_many(detector, sources: list[str | Path], confidence=config.DEFAULT_CONFIDENCE, classes=None, tiled=False) -> list[BatchItem]`.
  - `write_batch_csv(items: list[BatchItem], path: str | Path) -> Path`.

- [ ] **Bước 1: Viết test `tests/test_batch.py` (sẽ fail)**

```python
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
```

- [ ] **Bước 2:** Chạy `.venv/bin/pytest tests/test_batch.py -q`. Kỳ vọng: FAIL với `ImportError: cannot import name 'BatchItem'`.

- [ ] **Bước 3: Viết code**

Trong `vision_count/export.py`, đổi tên `_write_csv` thành `write_csv` ở định nghĩa và ở 2 chỗ gọi. Docstring: `"""Ghi CSV bằng utf-8-sig (có BOM) để Excel mở không lỗi dấu tiếng Việt."""`.

`vision_count/batch.py`:

```python
"""Đếm nhiều ảnh một lần và xuất bảng tổng hợp ra CSV."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vision_count import config
from vision_count.detector import DetectionResult, InvalidImageError, ObjectDetector
from vision_count.export import write_csv
from vision_count.labels_vi import vi_label


@dataclass
class BatchItem:
    """Kết quả của một ảnh: có result nếu đếm được, có error nếu file lỗi."""

    name: str
    result: DetectionResult | None = None
    error: str | None = None


def count_many(
    detector: ObjectDetector,
    sources: list[str | Path],
    confidence: float = config.DEFAULT_CONFIDENCE,
    classes: list[str] | None = None,
    tiled: bool = False,
) -> list[BatchItem]:
    """Đếm lần lượt từng ảnh. File hỏng được ghi lỗi, KHÔNG làm dừng các ảnh còn lại."""
    items = []
    for source in sources:
        name = Path(source).name
        try:
            items.append(BatchItem(name, detector.detect(source, confidence=confidence, classes=classes, tiled=tiled)))
        except InvalidImageError as exc:
            items.append(BatchItem(name, error=str(exc)))
    return items


def write_batch_csv(items: list[BatchItem], path: str | Path) -> Path:
    """Mỗi ảnh một dòng, mỗi loại vật một cột (tên tiếng Việt), cuối bảng có dòng tổng cộng."""
    path = Path(path)
    totals: Counter[str] = Counter()
    for item in items:
        if item.result:
            totals.update(item.result.counts)
    labels = [label for label, _ in totals.most_common()]  # loại nhiều nhất đứng đầu

    rows = []
    for item in items:
        if item.result:
            rows.append([item.name, item.result.total, *[item.result.counts.get(l, 0) for l in labels], ""])
        else:
            rows.append([item.name, "", *[""] * len(labels), item.error])
    rows.append(["Tổng cộng", sum(totals.values()), *[totals[l] for l in labels], ""])

    write_csv(path, ["Ảnh", "Tổng", *[vi_label(l) for l in labels], "Lỗi"], rows)
    return path
```

Trong `vision_count/__init__.py`, xuất thêm `BatchItem`, `count_many`, `write_batch_csv`.

- [ ] **Bước 4:** Chạy `.venv/bin/pytest -q`. Kỳ vọng: tất cả PASS.
- [ ] **Bước 5:** Commit `Thêm đếm nhiều ảnh: count_many và CSV tổng hợp`.

---

### Task 4: Lịch sử đếm (lõi, SQLite)

**Files:** Tạo `vision_count/history.py`. Sửa `vision_count/config.py`, `vision_count/__init__.py`, `.gitignore`. Test `tests/test_history.py`.

**Interfaces:**
- Produces:
  - `HistoryEntry(id: int, created_at: str, source: str, model_key: str, mode: str, total: int, counts: dict[str, int])`.
  - `HistoryStore(db_path: str | Path = config.HISTORY_DB)` với các phương thức:
    - `.add(source: str, model_key: str, mode: str, result: DetectionResult, created_at: datetime | None = None) -> int`
    - `.recent(limit: int = 50) -> list[HistoryEntry]`, mới nhất trước.
    - `.clear() -> None`
  - `config.HISTORY_DB = PROJECT_ROOT / "data" / "history.db"`.

- [ ] **Bước 1: Viết test `tests/test_history.py` (sẽ fail)**

```python
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
```

- [ ] **Bước 2:** Chạy `.venv/bin/pytest tests/test_history.py -q`. Kỳ vọng: FAIL với `ImportError: cannot import name 'HistoryStore'`.

- [ ] **Bước 3: Viết code**

Thêm vào cuối `vision_count/config.py`:

```python
# File SQLite lưu lịch sử đếm (chỉ số liệu, không lưu ảnh). Không đưa lên git.
HISTORY_DB = PROJECT_ROOT / "data" / "history.db"
```

Thêm `data/` vào `.gitignore` (kèm chú thích `# Lịch sử đếm trên máy`).

`vision_count/history.py`:

```python
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

from vision_count import config
from vision_count.detector import DetectionResult

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


@dataclass
class HistoryEntry:
    id: int
    created_at: str  # "YYYY-MM-DD HH:MM:SS"
    source: str  # tên file ảnh, hoặc "Camera"
    model_key: str
    mode: str  # mô tả chế độ, ví dụ "Thường", "Vật nhỏ, trong vùng"
    total: int
    counts: dict[str, int]


class HistoryStore:
    def __init__(self, db_path: str | Path = config.HISTORY_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self):
        # Mỗi lần dùng mở một kết nối mới rồi đóng ngay: đơn giản và an toàn khi Gradio chạy nhiều luồng.
        # closing() đóng kết nối; "with conn" bên trong tự commit (hoặc rollback nếu lỗi).
        return _Connection(self.db_path)

    def add(self, source: str, model_key: str, mode: str, result: DetectionResult, created_at: datetime | None = None) -> int:
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
                "SELECT id, created_at, source, model_key, mode, total, counts_json FROM history ORDER BY id DESC LIMIT ?",
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
```

Trong `vision_count/__init__.py`, xuất thêm `HistoryEntry`, `HistoryStore`.

- [ ] **Bước 4:** Chạy `.venv/bin/pytest -q`. Kỳ vọng: tất cả PASS.
- [ ] **Bước 5:** Commit `Thêm lịch sử đếm lưu bằng SQLite`.

---

### Task 5: Hàm xử lý giao diện (`ui/handlers.py`)

**Files:** Tạo `ui/handlers.py`. Test `tests/test_handlers.py`.

**Interfaces:**
- Consumes: mọi thứ từ Task 1-4; `new_export_dir` và các hằng số chuyển từ `ui/gradio_app.py` sang đây.
- Produces (để Task 6 nối vào giao diện):
  - `count_single(source_mode, image_path, webcam_image, model_key, confidence, selected_classes, label_style, tiled, region, history=None) -> (annotated, summary, counts_df, detail_df, files)`.
  - `add_region_point(image, points: list, region, xy) -> (image, points, region, info_text)`.
  - `clear_region() -> (None, [], info_text)`.
  - `count_batch(file_paths, model_key, confidence, selected_classes, label_style, tiled, history=None) -> (summary, table_df, gallery, csv_path)`.
  - `history_table(history, limit=50) -> pd.DataFrame`.
  - `clear_history(history) -> pd.DataFrame`.
  - Hằng số: `SOURCE_UPLOAD`, `SOURCE_WEBCAM`, `LABEL_STYLE_CHOICES`, `COUNT_COLUMNS`, `DETAIL_COLUMNS`, `BATCH_COLUMNS`, `HISTORY_COLUMNS`, `EXPORT_ROOT`, `MIN_REGION_SIZE = 5`, `NO_REGION_TEXT`.

- [ ] **Bước 1: Viết test `tests/test_handlers.py` (sẽ fail)**

```python
"""Test các hàm xử lý sự kiện của giao diện (gọi trực tiếp, không cần trình duyệt)."""

import os
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
    _, _, counts_table, detail_table, _ = _single()
    assert "người (person)" in counts_table["Loại vật thể"].tolist()
    assert "xe buýt (bus)" in detail_table["Loại vật thể"].tolist()


def test_returns_three_download_files_in_separate_folders():
    *_, first = _single()
    *_, second = _single()
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
    _, summary, counts_table, _, _ = _single(region=(0, 0, 400, 1080))  # nửa trái ảnh xe buýt
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

def test_add_region_point_two_clicks_make_region():
    img = Image.new("RGB", (200, 200))
    marked, points, region, info = handlers.add_region_point(img, [], None, (150, 120))
    assert points == [(150, 120)] and region is None and "điểm thứ 2" in info
    marked2, points2, region2, info2 = handlers.add_region_point(marked, points, None, (20, 30))
    assert points2 == [] and region2 == (20, 30, 150, 120)
    assert "Đếm" in info2


def test_add_region_point_rejects_tiny_region():
    img = Image.new("RGB", (200, 200))
    _, points, region, info = handlers.add_region_point(img, [(50, 50)], (1, 1, 100, 100), (52, 51))
    assert points == [] and region == (1, 1, 100, 100)  # giữ vùng cũ
    assert "quá nhỏ" in info


def test_add_region_point_without_image():
    out_img, points, region, info = handlers.add_region_point(None, [], None, (10, 10))
    assert out_img is None and points == [] and region is None
    assert "Đếm" in info


def test_clear_region():
    region, points, info = handlers.clear_region()
    assert region is None and points == [] and info == handlers.NO_REGION_TEXT


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
```

- [ ] **Bước 2:** Chạy `.venv/bin/pytest tests/test_handlers.py -q`. Kỳ vọng: FAIL với `ImportError: cannot import name 'handlers' from 'ui'`.

- [ ] **Bước 3: Viết `ui/handlers.py`**

```python
"""Các hàm xử lý sự kiện của giao diện Gradio.

Tách khỏi phần bố cục (gradio_app.py) để test được trực tiếp, không cần mở trình duyệt.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr
import pandas as pd

from vision_count import (
    LABEL_FULL,
    LABEL_NONE,
    LABEL_NUMBER,
    HistoryStore,
    InvalidImageError,
    config,
    count_many,
    display_label,
    draw_detections,
    draw_region,
    export_result,
    filter_by_region,
    get_detector,
    is_model_downloaded,
    load_image,
    normalize_region,
    vi_label,
    write_batch_csv,
)

COUNT_COLUMNS = ["Loại vật thể", "Số lượng"]
DETAIL_COLUMNS = ["#", "Loại vật thể", "Độ tin cậy", "Khung (x1, y1, x2, y2)"]
BATCH_COLUMNS = ["Ảnh", "Tổng", "Chi tiết"]
HISTORY_COLUMNS = ["Thời gian", "Nguồn", "Model", "Chế độ", "Tổng", "Chi tiết"]

# Hai nguồn ảnh: tải file lên hoặc chụp từ camera
SOURCE_UPLOAD = "upload"
SOURCE_WEBCAM = "webcam"

# Các kiểu nhãn cho người dùng chọn: (chữ hiển thị, giá trị)
LABEL_STYLE_CHOICES = [
    ("Đầy đủ: #1 người 87%", LABEL_FULL),
    ("Chỉ số thứ tự", LABEL_NUMBER),
    ("Chỉ khung", LABEL_NONE),
]

# Vùng đếm phải rộng/cao ít nhất 5px, tránh bấm 2 lần trùng một điểm tạo vùng rỗng
MIN_REGION_SIZE = 5
NO_REGION_TEXT = "Chưa chọn vùng: đếm cả ảnh. Muốn chỉ đếm một khu vực, bấm 2 góc lên ảnh kết quả."

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


def _model_name(model_key: str) -> str:
    return config.AVAILABLE_MODELS.get(model_key, (model_key,))[0]


def _load_model(model_key: str):
    """Nạp model; lỗi (sai tên, mất mạng khi tải lần đầu) thành thông báo trên giao diện."""
    if not is_model_downloaded(model_key):
        gr.Info(f"Đang tải model {_model_name(model_key)} lần đầu, vui lòng chờ...")
    try:
        return get_detector(model_key)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # Ví dụ: tải model thất bại vì mất mạng
        raise gr.Error(f"Không nạp được model {_model_name(model_key)}: {exc}") from exc


def _mode_text(tiled: bool, region) -> str:
    """Mô tả chế độ để hiện trong tóm tắt và lưu vào lịch sử."""
    parts = (["Vật nhỏ"] if tiled else []) + (["trong vùng đã chọn"] if region else [])
    return ", ".join(parts) or "Thường"


def _counts_text(counts: dict[str, int]) -> str:
    return ", ".join(f"{vi_label(label)}: {n}" for label, n in counts.items()) or "không có"


# ---------- Một ảnh ----------

def count_single(
    source_mode, image_path, webcam_image, model_key, confidence, selected_classes, label_style, tiled, region,
    history: HistoryStore | None = None,
):
    """Nút 'Đếm' của tab Một ảnh. Trả về 5 giá trị cho 5 ô kết quả."""
    if source_mode == SOURCE_WEBCAM:
        source, source_name = webcam_image, "Camera"  # Ảnh chụp từ camera, dạng mảng numpy
        if source is None:
            raise gr.Error("Vui lòng chụp ảnh từ camera trước.")
    else:
        source = image_path  # Đường dẫn file đã tải lên
        if not source:
            raise gr.Error("Vui lòng tải một ảnh lên trước.")
        source_name = Path(source).name

    detector = _load_model(model_key)
    try:
        image = load_image(source)
        result = detector.detect(image, confidence=confidence, classes=selected_classes or None, tiled=bool(tiled))
    except (InvalidImageError, ValueError) as exc:
        # gr.Error hiện thông báo lỗi màu đỏ trên giao diện thay vì làm sập app
        raise gr.Error(str(exc)) from exc

    result = filter_by_region(result, region)
    annotated = draw_detections(image, result.detections, label_fn=vi_label, label_style=label_style, region=region)
    mode = _mode_text(bool(tiled), region)

    footer = f"_Model: {_model_name(model_key)} · Chế độ: {mode}_"
    if result.total == 0:
        hint = " Vùng đã chọn có thể không chứa vật nào: bấm **Xóa vùng** để đếm cả ảnh." if region else ""
        summary = (
            "### Không tìm thấy vật thể nào\n"
            "Thử **giảm ngưỡng độ tin cậy**, bỏ bớt bộ lọc loại vật, hoặc dùng ảnh rõ hơn. "
            "Nếu vật bạn cần đếm không nằm trong 80 loại COCO, cần fine-tune model (giai đoạn sau)."
            f"{hint}\n\n{footer}"
        )
    else:
        summary = f"### Tổng cộng: {result.total} vật thể ({len(result.counts)} loại)\n{footer}"

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
    # Mỗi lần đếm ghi vào một thư mục riêng, để không ghi đè file của lần trước
    files = export_result(annotated, result, new_export_dir())
    if history is not None:
        history.add(source_name, model_key, mode, result)
    return annotated, summary, counts_table, detail_table, [str(p) for p in files]


# ---------- Chọn vùng bằng 2 lần bấm lên ảnh kết quả ----------

def add_region_point(image, points: list, region, xy):
    """Xử lý một lần bấm lên ảnh kết quả. Lần 1 đánh dấu góc thứ nhất, lần 2 tạo vùng.

    Trả về (ảnh xem trước, các điểm đang chờ, vùng, dòng hướng dẫn).
    """
    if image is None:
        return None, [], region, "Hãy bấm **Đếm** một ảnh trước, rồi bấm 2 góc lên ảnh kết quả để chọn vùng."
    x, y = (float(v) for v in xy)
    if not points:
        return draw_region(image, point=(x, y)), [(x, y)], region, f"Đã chọn góc 1 ({x:.0f}, {y:.0f}). Bấm điểm thứ 2."
    new_region = normalize_region(points[0], (x, y))
    if new_region[2] - new_region[0] < MIN_REGION_SIZE or new_region[3] - new_region[1] < MIN_REGION_SIZE:
        return image, [], region, "Vùng quá nhỏ, hãy bấm lại 2 góc cách xa nhau hơn."
    x1, y1, x2, y2 = new_region
    info = f"Vùng đếm: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f}). Bấm **Đếm** để chỉ đếm trong vùng này."
    return draw_region(image, region=new_region), [], new_region, info


def clear_region():
    """Nút 'Xóa vùng': quay lại đếm cả ảnh. Trả về (vùng, các điểm, dòng hướng dẫn)."""
    return None, [], NO_REGION_TEXT


# ---------- Nhiều ảnh ----------

def count_batch(file_paths, model_key, confidence, selected_classes, label_style, tiled, history: HistoryStore | None = None):
    """Nút 'Đếm tất cả' của tab Nhiều ảnh. Trả về (tóm tắt, bảng, gallery, file CSV)."""
    if not file_paths:
        raise gr.Error("Vui lòng chọn ít nhất một ảnh.")
    detector = _load_model(model_key)
    try:
        items = count_many(detector, file_paths, confidence=confidence, classes=selected_classes or None, tiled=bool(tiled))
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    mode = _mode_text(bool(tiled), None)
    gallery, rows = [], []
    for path, item in zip(file_paths, items):
        if item.result is None:
            rows.append([item.name, "", f"Lỗi: {item.error}"])
            continue
        rows.append([item.name, item.result.total, _counts_text(item.result.counts)])
        annotated = draw_detections(load_image(path), item.result.detections, label_fn=vi_label, label_style=label_style)
        gallery.append((annotated, f"{item.name}: {item.result.total}"))
        if history is not None:
            history.add(item.name, model_key, f"Nhiều ảnh, {mode}", item.result)

    ok = [i for i in items if i.result]
    errors = len(items) - len(ok)
    summary = (
        f"### {len(items)} ảnh, tổng cộng {sum(i.result.total for i in ok)} vật thể"
        + (f" ({errors} lỗi)" if errors else "")
        + f"\n_Model: {_model_name(model_key)} · Chế độ: {mode}_"
    )
    csv_path = write_batch_csv(items, new_export_dir() / "tong_hop_nhieu_anh.csv")
    return summary, pd.DataFrame(rows, columns=BATCH_COLUMNS), gallery, str(csv_path)


# ---------- Lịch sử ----------

def history_table(history: HistoryStore, limit: int = 50) -> pd.DataFrame:
    """Bảng các lần đếm gần nhất (mới nhất trước)."""
    rows = [
        [e.created_at, e.source, _model_name(e.model_key), e.mode, e.total, _counts_text(e.counts)]
        for e in history.recent(limit)
    ]
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def clear_history(history: HistoryStore) -> pd.DataFrame:
    """Nút 'Xóa lịch sử'."""
    history.clear()
    return history_table(history)
```

Ghi chú: summary của tab Nhiều ảnh phải chứa chữ "1 lỗi" khi có một file hỏng. Với chuỗi ở trên, kết quả là "(1 lỗi)", đúng như test kỳ vọng.

- [ ] **Bước 4:** Chạy `.venv/bin/pytest -q`. Kỳ vọng: tất cả PASS (`test_ui.py` cũ vẫn chạy với `gradio_app.py` cũ).
- [ ] **Bước 5:** Commit `Thêm ui/handlers.py: xử lý một ảnh, vùng đếm, nhiều ảnh, lịch sử`.

---

### Task 6: Bố cục giao diện 3 tab + README

**Files:** Sửa `ui/gradio_app.py` (viết lại bố cục), `tests/test_ui.py`, `README.md`.

**Interfaces:**
- Consumes: toàn bộ `ui/handlers.py` (Task 5); `HistoryStore` (Task 4).
- Produces: `build_app(history: HistoryStore | None = None) -> gr.Blocks`. `history=None` thì dùng `HistoryStore()` mặc định (`data/history.db`).

- [ ] **Bước 1: Viết lại `tests/test_ui.py` (sẽ fail)**

```python
"""Test dựng giao diện Gradio (các hàm xử lý được test ở test_handlers.py)."""

from vision_count import HistoryStore

from ui.gradio_app import build_app


def test_uploaded_files_are_kept_for_a_day(tmp_path):
    # Gradio xóa cả ảnh người dùng đã tải lên theo delete_cache; 1 giờ là quá ngắn khi tab vẫn mở
    _, max_age = build_app(HistoryStore(tmp_path / "h.db")).delete_cache
    assert max_age >= 86400


def test_app_has_three_main_tabs(tmp_path):
    app = build_app(HistoryStore(tmp_path / "h.db"))
    labels = {block.label for block in app.blocks.values() if type(block).__name__ == "Tab"}
    assert {"Một ảnh", "Nhiều ảnh", "Lịch sử"} <= labels
```

- [ ] **Bước 2:** Chạy `.venv/bin/pytest tests/test_ui.py -q`. Kỳ vọng: FAIL (`build_app()` chưa nhận tham số `history`, chưa có tab "Một ảnh").

- [ ] **Bước 3: Viết lại `ui/gradio_app.py`**

```python
"""Giao diện Gradio: chỉ bố cục và nối sự kiện. Xử lý nằm ở ui/handlers.py, AI nằm ở vision_count."""

from __future__ import annotations

import gradio as gr

from ui import handlers
from ui.handlers import (
    BATCH_COLUMNS,
    COUNT_COLUMNS,
    DETAIL_COLUMNS,
    HISTORY_COLUMNS,
    LABEL_STYLE_CHOICES,
    NO_REGION_TEXT,
    SOURCE_UPLOAD,
    SOURCE_WEBCAM,
)
from vision_count import LABEL_FULL, HistoryStore, config, display_label, get_detector


def build_app(history: HistoryStore | None = None) -> gr.Blocks:
    """Tạo giao diện. history=None thì lưu lịch sử vào data/history.db."""
    history = history or HistoryStore()

    # delete_cache: mỗi giờ Gradio xóa các file cũ hơn 1 ngày mà nó đã lưu, gồm bản sao file tải về
    # VÀ ảnh người dùng đã tải lên. Đặt 1 ngày (không phải 1 giờ) để ảnh không biến mất khi tab vẫn mở.
    with gr.Blocks(title="vision-count", delete_cache=(3600, 86400)) as app:
        gr.Markdown(
            "# vision-count: đếm vật thể trong ảnh\n"
            "Model nhận được 80 loại vật thông dụng (người, xe, chó, mèo, chai, cốc...)."
        )

        # ---------- Cài đặt dùng chung cho tab Một ảnh và Nhiều ảnh ----------
        with gr.Accordion("Cài đặt", open=True):
            with gr.Row():
                model_input = gr.Dropdown(
                    choices=[(label, key) for key, (label, _) in config.AVAILABLE_MODELS.items()],
                    value=config.DEFAULT_MODEL_KEY,
                    label="Model",
                    info="Small chính xác hơn với vật nhỏ/bị che. Lần đầu chọn sẽ tải file (~19MB).",
                )
                confidence_input = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    step=0.05,
                    value=config.DEFAULT_CONFIDENCE,
                    label="Ngưỡng độ tin cậy",
                    info="Cao: ít khung hơn nhưng chắc chắn hơn. Thấp: bắt được nhiều hơn nhưng dễ nhầm.",
                )
                tiled_input = gr.Checkbox(
                    value=False,
                    label="Chế độ vật nhỏ",
                    info="Chia ảnh thành ô 640px để bắt vật nhỏ/ở xa. Chậm hơn, hợp với ảnh lớn.",
                )
            with gr.Row():
                class_input = gr.Dropdown(
                    # (chữ hiển thị, giá trị gửi về): hiện "người (person)" nhưng gửi về "person"
                    choices=[(display_label(name), name) for name in get_detector().class_names],
                    multiselect=True,
                    label="Chỉ đếm các loại (để trống = đếm tất cả)",
                )
                label_style_input = gr.Radio(
                    choices=LABEL_STYLE_CHOICES,
                    value=LABEL_FULL,
                    label="Kiểu nhãn trên ảnh",
                    info="Khi có nhiều vật, chọn kiểu gọn để nhãn không đè lên nhau.",
                )

        with gr.Tabs():
            # ---------- Tab Một ảnh ----------
            with gr.Tab("Một ảnh"):
                # Ghi nhớ tab nguồn đang mở, vùng đếm đã chọn và góc đang chờ góc thứ 2
                source_mode = gr.State(SOURCE_UPLOAD)
                region_state = gr.State(None)
                points_state = gr.State([])

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Tabs():
                            with gr.Tab("Tải ảnh") as upload_tab:
                                # Dùng gr.File thay vì gr.Image để chính code của mình kiểm tra file,
                                # nhờ vậy file hỏng hoặc không phải ảnh sẽ có thông báo lỗi tiếng Việt rõ ràng.
                                image_input = gr.File(label="Ảnh cần đếm", file_types=["image"], type="filepath")
                            with gr.Tab("Camera") as webcam_tab:
                                # sources=["webcam"]: chỉ cho chụp từ camera. Trình duyệt sẽ hỏi quyền lần đầu.
                                webcam_input = gr.Image(label="Chụp ảnh từ camera", sources=["webcam"], type="numpy")
                        # Khi người dùng bấm chuyển tab thì cập nhật source_mode
                        upload_tab.select(fn=lambda: SOURCE_UPLOAD, outputs=source_mode)
                        webcam_tab.select(fn=lambda: SOURCE_WEBCAM, outputs=source_mode)

                        region_info = gr.Markdown(NO_REGION_TEXT)
                        clear_region_button = gr.Button("Xóa vùng", size="sm")
                        run_button = gr.Button("Đếm", variant="primary")

                    with gr.Column(scale=2):
                        image_output = gr.Image(label="Kết quả (bấm 2 góc lên ảnh để chọn vùng đếm)", type="pil")
                        summary_output = gr.Markdown()
                        counts_output = gr.Dataframe(headers=COUNT_COLUMNS, label="Số lượng theo loại", interactive=False)
                        with gr.Accordion("Chi tiết từng khung", open=False):
                            detail_output = gr.Dataframe(headers=DETAIL_COLUMNS, interactive=False)
                        download_output = gr.File(
                            label="Tải kết quả về (ảnh + CSV)", file_count="multiple", interactive=False
                        )

            # ---------- Tab Nhiều ảnh ----------
            with gr.Tab("Nhiều ảnh"):
                batch_input = gr.File(label="Chọn nhiều ảnh", file_types=["image"], file_count="multiple", type="filepath")
                batch_button = gr.Button("Đếm tất cả", variant="primary")
                batch_summary = gr.Markdown()
                batch_table = gr.Dataframe(headers=BATCH_COLUMNS, label="Kết quả từng ảnh", interactive=False)
                batch_download = gr.File(label="Tải bảng tổng hợp (CSV)", interactive=False)
                batch_gallery = gr.Gallery(label="Ảnh đã khoanh khung", columns=3, height="auto")

            # ---------- Tab Lịch sử ----------
            with gr.Tab("Lịch sử") as history_tab:
                gr.Markdown("Các lần đếm gần nhất (lưu trên máy, chỉ số liệu, không lưu ảnh).")
                with gr.Row():
                    refresh_button = gr.Button("Làm mới", size="sm")
                    clear_history_button = gr.Button("Xóa lịch sử", size="sm", variant="stop")
                history_output = gr.Dataframe(headers=HISTORY_COLUMNS, interactive=False)

        # ---------- Nối sự kiện ----------
        settings = [model_input, confidence_input, class_input, label_style_input, tiled_input]

        def on_count(mode, path, frame, model_key, conf, classes, style, tiled, region):
            return handlers.count_single(mode, path, frame, model_key, conf, classes, style, tiled, region, history)

        run_button.click(
            fn=on_count,
            inputs=[source_mode, image_input, webcam_input, *settings[:4], tiled_input, region_state],
            outputs=[image_output, summary_output, counts_output, detail_output, download_output],
        )

        def on_image_click(image, points, region, evt: gr.SelectData):
            # evt.index = [x, y]: tọa độ pixel trên ảnh gốc nơi người dùng bấm
            return handlers.add_region_point(image, points, region, evt.index)

        image_output.select(
            fn=on_image_click,
            inputs=[image_output, points_state, region_state],
            outputs=[image_output, points_state, region_state, region_info],
        )
        clear_region_button.click(fn=handlers.clear_region, outputs=[region_state, points_state, region_info])

        def on_batch(paths, model_key, conf, classes, style, tiled):
            return handlers.count_batch(paths, model_key, conf, classes, style, tiled, history)

        batch_button.click(
            fn=on_batch,
            inputs=[batch_input, *settings],
            outputs=[batch_summary, batch_table, batch_gallery, batch_download],
        )

        show_history = lambda: handlers.history_table(history)  # noqa: E731
        history_tab.select(fn=show_history, outputs=history_output)
        refresh_button.click(fn=show_history, outputs=history_output)
        clear_history_button.click(fn=lambda: handlers.clear_history(history), outputs=history_output)

    return app
```

- [ ] **Bước 4:** Chạy `.venv/bin/pytest -q`. Kỳ vọng: tất cả PASS.

- [ ] **Bước 5: Cập nhật README**
  - Cây thư mục: thêm `region.py`, `tiling.py`, `batch.py`, `history.py`, `ui/handlers.py`, `data/` (lịch sử, không đưa lên git).
  - Mục "Cách dùng": mô tả bố cục mới (Cài đặt ở trên, 3 tab); cách chọn **vùng đếm** (đếm trước, bấm 2 góc lên ảnh kết quả, bấm Đếm lại; quy tắc tính theo tâm khung; nút Xóa vùng); **Chế độ vật nhỏ** (khi nào dùng, chậm hơn); tab **Nhiều ảnh** (CSV tổng hợp, file lỗi ghi vào bảng); tab **Lịch sử** (chỉ lưu số liệu, `data/history.db`, nút Xóa lịch sử).
  - Mục 5 (code): ví dụ `detect(..., tiled=True)`, `filter_by_region`, `count_many` + `write_batch_csv`, `HistoryStore().recent()`.

- [ ] **Bước 6: Kiểm tra trên trình duyệt** (`.venv/bin/python app.py`)
  - Tab Một ảnh: tải `samples/bus.jpg`, bấm Đếm, bấm 2 góc bao nửa trái ảnh → hiện khung vàng và dòng "Vùng đếm: ...". Bấm Đếm → số lượng giảm, tóm tắt ghi "trong vùng đã chọn". Bấm Xóa vùng → Đếm lại ra đủ 5.
  - Bật Chế độ vật nhỏ → Đếm → tóm tắt ghi "Vật nhỏ".
  - Tab Nhiều ảnh: chọn `bus.jpg` + `zidane.jpg` → bảng 2 dòng, gallery 2 ảnh, tải được CSV.
  - Tab Lịch sử: thấy các lần đếm vừa rồi; Xóa lịch sử → bảng trống.

- [ ] **Bước 7:** Commit `Giao diện 3 tab: một ảnh (vùng đếm, vật nhỏ), nhiều ảnh, lịch sử`.
