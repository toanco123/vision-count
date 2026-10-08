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
