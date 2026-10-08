"""Điểm khởi động ứng dụng. Chạy: python app.py"""

from ui.gradio_app import build_app
from vision_count import get_detector


def main():
    print("Đang nạp model YOLO (lần đầu sẽ tải file model về, mất vài giây)...")
    get_detector()  # Nạp sẵn model mặc định để lần bấm "Đếm" đầu tiên không phải chờ
    app = build_app()
    # 127.0.0.1 = chỉ máy bạn truy cập được. Mở trình duyệt tại http://127.0.0.1:7860
    app.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
