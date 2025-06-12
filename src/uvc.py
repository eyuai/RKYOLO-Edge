import re
import subprocess
import os
import traceback

# import cv2
from utils import logger
from linuxpy.video.device import Device


def list_video_devices():
    """列出所有 /dev/video* 设备编号"""
    devices = []
    for dev in os.listdir("/dev"):
        if dev.startswith("video") and dev[5:].isdigit():
            devices.append(int(dev[5:]))
    return sorted(devices)


def get_supported_formats(device_path):
    """使用 v4l2-ctl 获取摄像头支持的格式和分辨率"""
    if not os.path.exists(device_path):
        logger.error(f"Device not found: {device_path}")
        return None
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--list-formats-ext"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to list formats: {e}")
        return None


def parse_MJPG_resolutions_and_fps(formats_output):
    """解析 v4l2-ctl 输出以获取 MJPG 格式下的分辨率及帧率"""
    lines = formats_output.split("\n")
    in_MJPG_section = False
    current_resolution = None
    resolution_fps_map = {}

    for line in lines:
        line = line.strip()
        if "'MJPG'" in line or "MJPG" in line:
            in_MJPG_section = True
            print("Found MJPG section start")
            continue
        if in_MJPG_section and line.startswith("Size: Discrete"):
            match = re.search(r"Size: Discrete (\d+)x(\d+)", line)
            if match:
                width, height = map(int, match.groups())
                current_resolution = (width, height)
                resolution_fps_map[current_resolution] = []
        elif in_MJPG_section and line.startswith("Interval:"):
            match = re.search(r"\(([\d.]+) fps\)", line)
            if match and current_resolution:
                fps = float(match.group(1))
                resolution_fps_map[current_resolution].append(fps)
        elif in_MJPG_section and (
            "Pixel Format" in line
            or line.startswith("[")
            and not line.startswith("[0]")
        ):
            # 遇到新的Pixel Format或者新的格式序号，结束当前MJPG区块
            in_MJPG_section = False
            current_resolution = None

    if not resolution_fps_map:
        return None, None
    max_resolution = max(resolution_fps_map.keys(), key=lambda x: x[0] * x[1])
    max_fps = max(resolution_fps_map[max_resolution])

    print(f"Max resolution: {max_resolution}, max fps: {max_fps}")
    return max_resolution, max_fps


def set_camera_format_v4l2(dev_path, width, height, pix_fmt, fps):
    try:
        print(
            f"Setting format on {dev_path}: {width}x{height} pix_fmt={pix_fmt} fps={int(fps)}"
        )
        subprocess.run(
            [
                "v4l2-ctl",
                "-d",
                dev_path,
                "--set-fmt-video",
                f"width={width},height={height},pixelformat={pix_fmt}",
            ],
            check=True,
        )
        subprocess.run(
            ["v4l2-ctl", "-d", dev_path, f"--set-parm={int(fps)}"], check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to set format: {e}")
        logger.error(traceback.format_exc())


def camera_init_v4l2(video_source=None):
    """
    初始化摄像头为 MJPG 模式并选用最大分辨率。
    如果 video_source 为 None，则自动扫描第一个可用摄像头。
    """
    sources_to_try = []

    if video_source is None:
        device_ids = list_video_devices()
        sources_to_try = [f"/dev/video{n}" for n in device_ids]
    elif isinstance(video_source, int):
        sources_to_try = [f"/dev/video{video_source}"]
    elif isinstance(video_source, str):
        sources_to_try = [video_source]
    else:
        logger.error(f"Invalid video source type: {type(video_source)}")
        return None, None, None

    for device_path in sources_to_try:
        if not os.path.exists(device_path):
            continue

        formats_output = get_supported_formats(device_path)
        if not formats_output:
            continue

        max_resolution, max_fps = parse_MJPG_resolutions_and_fps(formats_output)
        if not max_resolution or not max_fps:
            continue

        logger.warning(
            f"Trying {device_path} with resolution: {max_resolution[0]}x{max_resolution[1]} max fps:{max_fps}"
        )

        set_camera_format_v4l2(
            device_path, max_resolution[0], max_resolution[1], "MJPG", max_fps
        )
        # cap = Device.from_id(device_path)

        # if cap:
        return device_path, max_resolution[0], max_resolution[1]

    logger.error("No usable camera found with MJPG support")
    return None, None, None
