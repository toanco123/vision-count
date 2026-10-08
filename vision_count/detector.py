"""Module phát hiện và đếm vật thể bằng YOLO (Ultralytics).

File này KHÔNG import Gradio hay FastAPI. Nó chỉ nhận ảnh vào và trả về dữ liệu
thuần Python, nhờ vậy giao diện nào cũng dùng lại được.
"""

from __future__ import annotations

import io
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from vision_count import config

# Các kiểu "ảnh đầu vào" mà module chấp nhận
ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]


class InvalidImageError(ValueError):
    """Lỗi khi đầu vào không phải là ảnh hợp lệ."""


@dataclass
class Detection:
    """Một vật thể được phát hiện."""

    label: str  # Tên loại vật, ví dụ "person"
    class_id: int  # Mã số loại vật trong model (COCO: 0..79)
    confidence: float  # Độ tin cậy 0..1
    box: tuple[float, float, float, float]  # Khung (x1, y1, x2, y2) theo pixel


@dataclass
class DetectionResult:
    """Kết quả phát hiện trên một ảnh."""

    detections: list[Detection] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)  # {"person": 3, "car": 2}
    image_width: int = 0
    image_height: int = 0

    @property
    def total(self) -> int:
        """Tổng số vật thể đếm được (mọi loại)."""
        return len(self.detections)

    def to_dict(self) -> dict:
        """Chuyển sang dict để trả về JSON (dùng cho FastAPI sau này)."""
        data = asdict(self)
        data["total"] = self.total
        return data


def load_image(source: ImageInput, name: str | None = None) -> Image.Image:
    """Đọc ảnh từ nhiều dạng đầu vào và trả về ảnh PIL hệ màu RGB.

    name: tên hiển thị trong thông báo lỗi (vd tên file người dùng tải lên qua API).
    Ném InvalidImageError nếu đầu vào không phải ảnh hoặc ảnh bị hỏng.
    """
    if source is None:
        raise InvalidImageError("Chưa có ảnh nào được tải lên.")

    # Trường hợp đã là ảnh trong bộ nhớ
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, np.ndarray):
        if source.ndim not in (2, 3) or source.size == 0:
            raise InvalidImageError("Mảng dữ liệu không phải ảnh hợp lệ.")
        return Image.fromarray(source).convert("RGB")

    # Trường hợp là đường dẫn file hoặc dữ liệu bytes (file tải lên qua API)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise InvalidImageError(f"Không tìm thấy file: {path.name}")
        opener = lambda: Image.open(path)  # noqa: E731
        name = name or path.name
    elif isinstance(source, bytes):
        opener = lambda: Image.open(io.BytesIO(source))  # noqa: E731
        name = name or "dữ liệu tải lên"
    else:
        raise InvalidImageError(f"Không hỗ trợ kiểu dữ liệu: {type(source).__name__}")

    try:
        # Bước 1: verify() kiểm tra file có đúng là ảnh và không bị hỏng
        with opener() as img:
            img.verify()
        # Bước 2: verify() làm đối tượng không dùng được nữa, nên phải mở lại.
        with opener() as img:
            # Ảnh chụp từ điện thoại hay bị xoay; exif_transpose xoay lại cho đúng chiều
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError(
            f"'{name}' không phải là file ảnh hợp lệ (hỗ trợ JPG, PNG, WEBP, BMP...)."
        ) from exc


class ObjectDetector:
    """Bọc model YOLO: nạp model một lần, gọi detect() nhiều lần."""

    def __init__(self, model_path: str | Path = config.DEFAULT_MODEL, device: str | None = None):
        # Import ở đây để việc import package nhanh, chỉ nạp PyTorch khi thật sự cần
        from ultralytics import YOLO

        self.model_path = self._resolve_model_path(model_path)
        self.model = YOLO(str(self.model_path))
        self.device = device  # None = để Ultralytics tự chọn (CPU trên máy không có GPU)

    @staticmethod
    def _resolve_model_path(model_path: str | Path) -> Path:
        """Nếu chỉ truyền tên file (vd 'yolo11n.pt') thì lưu/tìm trong thư mục models/."""
        path = Path(model_path)
        if path.parent == Path("."):
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            return config.MODELS_DIR / path.name
        return path

    @property
    def class_names(self) -> list[str]:
        """Danh sách tên các loại vật model nhận được (COCO có 80 loại)."""
        return [self.model.names[i] for i in sorted(self.model.names)]

    def detect(
        self,
        image: ImageInput,
        confidence: float = config.DEFAULT_CONFIDENCE,
        classes: list[str] | None = None,
        tiled: bool = False,
    ) -> DetectionResult:
        """Phát hiện vật thể trong ảnh.

        Args:
            image: ảnh đầu vào (đường dẫn, bytes, ảnh PIL hoặc mảng numpy).
            confidence: ngưỡng độ tin cậy 0..1. Khung thấp hơn ngưỡng sẽ bị bỏ.
            classes: chỉ đếm các loại này (vd ["person", "car"]). None = đếm tất cả.
            tiled: chế độ vật nhỏ, chia ảnh thành ô 640px (chậm hơn).
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Ngưỡng độ tin cậy phải nằm trong khoảng 0 đến 1.")

        pil_image = load_image(image)
        class_ids = self._class_names_to_ids(classes) if classes else None

        if tiled:
            # import ở đây để tránh import vòng (tiling.py import Detection từ file này)
            from vision_count.tiling import drop_cut_boxes, make_tiles, merge_overlaps

            w, h = pil_image.width, pil_image.height
            tiles = make_tiles(w, h, config.DEFAULT_IMAGE_SIZE, config.TILE_OVERLAP)
            # Lượt 0: cả ảnh (bắt vật to). Các lượt sau: từng ô (bắt vật nhỏ), bỏ khung bị ô cắt dở.
            detections = self._predict(pil_image, confidence, class_ids)
            if len(tiles) > 1:
                passes = [detections]
                for x1, y1, x2, y2 in tiles:
                    tile_dets = self._predict(pil_image.crop((x1, y1, x2, y2)), confidence, class_ids, x1, y1)
                    passes.append(drop_cut_boxes(tile_dets, (x1, y1, x2, y2), w, h))
                detections = merge_overlaps(passes)
        else:
            detections = self._predict(pil_image, confidence, class_ids)

        # Sắp xếp khung theo độ tin cậy giảm dần
        detections.sort(key=lambda d: d.confidence, reverse=True)
        # Đếm số lượng theo từng loại, loại nhiều nhất đứng đầu
        counts = dict(Counter(d.label for d in detections).most_common())

        return DetectionResult(
            detections=detections,
            counts=counts,
            image_width=pil_image.width,
            image_height=pil_image.height,
        )

    def _predict(
        self,
        image: Image.Image,
        confidence: float,
        class_ids: list[int] | None,
        offset_x: float = 0,
        offset_y: float = 0,
    ) -> list[Detection]:
        """Chạy model trên một ảnh; cộng offset để đổi tọa độ trong ô về tọa độ ảnh gốc."""
        # verbose=False để không in log ra màn hình mỗi lần.
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
        for xyxy, conf, cls in zip(
            boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
        ):
            class_id = int(cls)
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            detections.append(
                Detection(
                    label=self.model.names[class_id],
                    class_id=class_id,
                    confidence=round(float(conf), 4),
                    box=(
                        round(x1 + offset_x, 1),
                        round(y1 + offset_y, 1),
                        round(x2 + offset_x, 1),
                        round(y2 + offset_y, 1),
                    ),
                )
            )
        return detections

    def _class_names_to_ids(self, names: list[str]) -> list[int]:
        """Đổi tên loại vật ('person') sang mã số (0) mà YOLO cần."""
        name_to_id = {name: idx for idx, name in self.model.names.items()}
        unknown = [n for n in names if n not in name_to_id]
        if unknown:
            raise ValueError(f"Model không biết các loại vật: {', '.join(unknown)}")
        return [name_to_id[n] for n in names]
