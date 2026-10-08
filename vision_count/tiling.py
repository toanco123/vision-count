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


def drop_cut_boxes(
    detections: list[Detection], tile: tuple[int, int, int, int], width: int, height: int, margin: float = 2
) -> list[Detection]:
    """Bỏ các khung chạm mép BÊN TRONG của ô (mép không trùng mép ảnh): đó là vật bị ô cắt dở.

    Vật nhỏ nằm ở mép ô sẽ hiện nguyên vẹn ở ô bên cạnh (nhờ phần chồng 20%);
    vật to vắt qua nhiều ô thì đã có lượt nhận diện cả ảnh lo.
    """
    x1, y1, x2, y2 = tile
    kept = []
    for d in detections:
        bx1, by1, bx2, by2 = d.box
        cut = (
            (x1 > 0 and bx1 <= x1 + margin)
            or (y1 > 0 and by1 <= y1 + margin)
            or (x2 < width and bx2 >= x2 - margin)
            or (y2 < height and by2 >= y2 - margin)
        )
        if not cut:
            kept.append(d)
    return kept


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Phần giao chia cho phần hợp của 2 khung (0 = không chạm, 1 = trùng khít)."""
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _area(box: tuple[float, ...]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _ios(small: tuple[float, ...], big: tuple[float, ...]) -> float:
    """Tỉ lệ diện tích khung `small` nằm bên trong khung `big` (1 = nằm trọn bên trong)."""
    inter_w = min(small[2], big[2]) - max(small[0], big[0])
    inter_h = min(small[3], big[3]) - max(small[1], big[1])
    if inter_w <= 0 or inter_h <= 0 or _area(small) == 0:
        return 0.0
    return inter_w * inter_h / _area(small)


def merge_overlaps(
    passes: list[list[Detection]], iou_threshold: float = 0.5, fragment_threshold: float = 0.8
) -> list[Detection]:
    """Gộp khung trùng giữa các LƯỢT nhận diện (lượt 0 = cả ảnh, các lượt sau = từng ô).

    - Khung của lượt cả ảnh luôn được giữ (YOLO đã tự lọc trùng trong một lượt).
    - Khung của một ô bị bỏ nếu cùng loại với một khung đã giữ ở LƯỢT KHÁC và:
        * gần trùng khít với nó (IoU > iou_threshold): cùng một vật được thấy 2 lần; hoặc
        * nằm gần trọn bên trong một khung to hơn của lượt cả ảnh (> fragment_threshold):
          mảnh vụn của vật to, ví dụ ô chỉ thấy đôi chân của một người rất to.
    - Hai khung trong CÙNG một lượt không bao giờ bị gộp, nên trẻ em đứng trước người lớn
      (khung nằm trong khung, cùng một ô) vẫn được đếm đủ.
    """
    kept: list[tuple[Detection, int]] = [(det, 0) for det in passes[0]] if passes else []
    tile_candidates = [(det, i) for i, dets in enumerate(passes[1:], start=1) for det in dets]
    tile_candidates.sort(key=lambda c: c[0].confidence, reverse=True)
    for det, pass_id in tile_candidates:
        duplicate = any(
            k.class_id == det.class_id
            and k_pass != pass_id
            and (
                _iou(k.box, det.box) > iou_threshold
                or (k_pass == 0 and _area(k.box) >= _area(det.box) and _ios(det.box, k.box) > fragment_threshold)
            )
            for k, k_pass in kept
        )
        if not duplicate:
            kept.append((det, pass_id))
    return sorted((det for det, _ in kept), key=lambda d: d.confidence, reverse=True)
