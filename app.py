"""Điểm khởi động ứng dụng. Chạy: python app.py"""

from ui.gradio_app import build_app
from vision_count import ObjectDetector


def main():
    print("Đang nạp model YOLO (lần đầu sẽ tải file model về, mất vài giây)...")
    detector = ObjectDetector()
    app = build_app(detector)
    # 127.0.0.1 = chỉ máy bạn truy cập được. Mở trình duyệt tại http://127.0.0.1:7860
    app.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
