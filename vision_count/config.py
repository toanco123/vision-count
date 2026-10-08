"""Cấu hình mặc định của project. Sửa ở đây thay vì sửa rải rác trong code."""

from pathlib import Path

# Thư mục gốc của project (thư mục chứa app.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Thư mục lưu file trọng số (weights) của model
MODELS_DIR = PROJECT_ROOT / "models"

# Model YOLO bản nano: nhỏ (~5MB), chạy được trên CPU.
# Lần chạy đầu Ultralytics sẽ tự tải file này về (chỉ tải 1 lần).
# Sau này fine-tune xong, chỉ cần đổi thành đường dẫn tới file best.pt của bạn.
DEFAULT_MODEL = "yolo11n.pt"

# Ngưỡng độ tin cậy mặc định: chỉ giữ các khung model "chắc chắn" >= 25%
DEFAULT_CONFIDENCE = 0.25

# Kích thước ảnh đưa vào model (pixel). 640 là chuẩn của YOLO.
DEFAULT_IMAGE_SIZE = 640
