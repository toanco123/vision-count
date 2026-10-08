"""Cấu hình mặc định của project. Sửa ở đây thay vì sửa rải rác trong code."""

from pathlib import Path

# Thư mục gốc của project (thư mục chứa app.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Thư mục lưu file trọng số (weights) của model
MODELS_DIR = PROJECT_ROOT / "models"

# Model YOLO bản nano: nhỏ (~5MB), chạy được trên CPU.
# Lần chạy đầu Ultralytics sẽ tự tải file này về (chỉ tải 1 lần).
# Đây là model mặc định khi tự tạo ObjectDetector() trong code.
# Giao diện lấy danh sách model từ AVAILABLE_MODELS bên dưới: fine-tune xong thì thêm
# một dòng vào đó, ví dụ "cua_toi": ("Model của tôi", "best.pt").
DEFAULT_MODEL = "yolo11n.pt"

# Ngưỡng độ tin cậy mặc định: chỉ giữ các khung model "chắc chắn" >= 25%
DEFAULT_CONFIDENCE = 0.25

# Kích thước ảnh đưa vào model (pixel). 640 là chuẩn của YOLO.
DEFAULT_IMAGE_SIZE = 640

# Các model cho người dùng chọn: khóa -> (tên hiển thị, file weights).
# Bản small chính xác hơn nhưng chậm hơn; lần đầu chọn sẽ tải file (~19MB).
AVAILABLE_MODELS = {
    "nano": ("Nano: nhanh nhất", "yolo11n.pt"),
    "small": ("Small: chính xác hơn, chậm hơn", "yolo11s.pt"),
}
DEFAULT_MODEL_KEY = "nano"

# Chế độ vật nhỏ: các ô cạnh nhau chồng lên nhau 20% để vật nằm ở mép ô không bị bỏ sót
TILE_OVERLAP = 0.2
