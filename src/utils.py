import hashlib
import socket
import logging
import numpy as np


def iou(box1, box2):
    """
    计算两个边界框的交并比（IoU）
    :param box1: 边界框1, [x1_min, y1_min, x1_max, y1_max]
    :param box2: 边界框2, [x2_min, y2_min, x2_max, y2_max]
    :return: IoU值
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 计算交集
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 计算并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    # 计算交并比
    return inter_area / union_area if union_area > 0 else 0


def iou_xyhw(box1, box2):
    """
    计算两个边界框的交并比（IoU）
    :return: IoU值
    """
    x1_min = box1["x"]
    y1_min = box1["y"]
    x1_max = box1["x"] + box1["width"]
    y1_max = box1["y"] + box1["height"]
    x2_min = box2["x"]
    y2_min = box2["y"]
    x2_max = box2["x"] + box2["width"]
    y2_max = box2["y"] + box2["height"]

    # 计算交集
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 计算并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    # 计算交并比
    return inter_area / union_area if union_area > 0 else 0


# 获取内网IP
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# 获取CPU序列号
def get_machine_id():
    # 尝试读取 CPU 序列号（适用于树莓派等 ARM 设备）
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    return line.split(":")[1].strip()
    except FileNotFoundError:
        pass

    # 尝试读取 `/etc/machine-id`
    try:
        with open("/etc/machine-id", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        pass

    # 尝试读取 `/var/lib/dbus/machine-id`（某些 Linux 发行版使用）
    try:
        with open("/var/lib/dbus/machine-id", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        pass

    return "0000000000000000"  # 如果都获取不到，返回默认值


# 生成哈希密码
def get_password_hash(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Logging utility
def log(message: str, level="INFO"):
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
    elif level == "DEBUG":
        logging.debug(message)
    else:
        logging.info(message)
    print(f"[{level}] {message}")  # 仍然输出到控制台


def is_decoded_image(img):
    # 判断是不是多维数组，且至少二维以上
    if not isinstance(img, np.ndarray):
        return False
    if img.ndim < 2:
        return False
    if img.dtype != np.uint8:
        return False
    # 可以简单用形状判断，彩色图一般是3维，灰度图一般2维
    if img.ndim == 2 or img.ndim == 3:
        return True
    return False


# 创建日志器
logger = logging.getLogger("camera-detector")
logger.setLevel(logging.INFO)
