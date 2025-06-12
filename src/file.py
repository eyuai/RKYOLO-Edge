import aiohttp
import asyncio
from utils import logger
import json
import redis
import time
import hashlib, threading
import os
from db import RedisDetectionDB
from utils import iou_xyhw as iou

timeout_settings = aiohttp.ClientTimeout(
    total=1,  # 总超时时间
    connect=1,  # 连接超时（单位：秒）
    sock_connect=15,  # TCP 连接超时
    sock_read=15,  # 读取响应数据超时
)


def calculate_md5(file_content):
    """计算文件内容的 MD5 值"""
    md5_hash = hashlib.md5()
    md5_hash.update(file_content)
    return md5_hash.hexdigest()


async def upload_file(url, files, frame_num, rcli: RedisDetectionDB = None):
    # 设置超时时间
    global timeout_settings
    response_status = None
    start_time = time.time()  # 记录请求开始时间

    async with aiohttp.ClientSession(timeout=timeout_settings) as session:
        try:
            # 提取 detectionDatas 并放入 headers
            detection_datas = files["detectionDatas"][1]
            headers = {"detection_datas": detection_datas}

            # 构建 multipart 表单数据
            data = aiohttp.FormData()
            for key, value in files.items():
                data.add_field(key, value[1], filename=value[0], content_type=value[2])

            # 异步发送 POST 请求
            async with session.post(url, headers=headers, data=data) as response:
                if response.status == 200:
                    # logger.error(f"Upload successful :{detection_datas}")
                    response_status = 1
                else:
                    logger.error(f"Upload failed with status code {response.status}")
                    response_status = 0
        except asyncio.TimeoutError:
            logger.debug(f"Upload request timed out:{detection_datas}")
            response_status = 0
        except Exception as e:
            response_status = 0
            logger.error(f"Error during upload: {e}")
    if rcli:
        rcli.redis_client.rpush(
            "upload_results", str({"frame_num": frame_num, "status": response_status})
        )
    else:
        with redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        ) as redis_client:
            redis_client.rpush(
                "upload_results",
                str({"frame_num": frame_num, "status": response_status}),
            )
    end_time = time.time()  # 记录请求结束时间
    request_time = end_time - start_time  # 计算请求时间
    logger.debug(f"Request time: {request_time:.2f} seconds")


def upload_file_sync(url, files):
    """同步文件上传函数"""
    try:
        data = aiohttp.FormData()
        for key, value in files.items():
            data.add_field(key, value[1], filename=value[0], content_type=value[2])

        with aiohttp.ClientSession(timeout=timeout_settings) as session:
            response = session.post(url, data=data)
            # del data
            if response.status == 200:
                logger.info("Upload successful")
            else:
                logger.error(f"Upload failed with status code {response.status}")
    except Exception as e:
        logger.error(f"Error during upload: {e}")


def run_upload(url, files):
    """在新线程中运行 asyncio 事件循环"""
    asyncio.run(upload_file(url, files))


# 保存文件到本地
def save_file(file_path, file_content):
    """保存文件到本地"""
    with open(file_path, "wb") as f:
        f.write(file_content)
    logger.debug(f"File saved to {file_path}")


# 删除文件
def delete_file(file_path):
    """删除文件"""
    try:
        os.remove(file_path)
        logger.debug(f"File deleted: {file_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file: {e}")


# 读取文件
def read_file(file_path):
    """读取文件内容"""
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return None


def calculate_iou(last_boxes, current_boxes):
    """计算 IOU"""
    # 对比数组长度
    if len(last_boxes) != len(current_boxes):
        return True
    # 排序boxes

    last_boxes = sorted(last_boxes, key=lambda x: x["x"])
    current_boxes = sorted(current_boxes, key=lambda x: x["x"])
    for index in range(len(last_boxes)):
        last_box = last_boxes[index]
        current_box = current_boxes[index]
        iou_res = iou(last_box, current_box)
        if iou_res > 0.5:
            return True
    return False  # IOU阈值


class AsyncUploader:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def upload(self, coro, on_done=None):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        if on_done:

            def done_callback(fut):
                try:
                    fut.result()  # 确保抛出异常被处理
                    on_done()
                except Exception as e:
                    print(f"上传任务失败: {e}")

            future.add_done_callback(done_callback)


# 循环从数据库读取文件数据，查看redis 的iou，iou不同更新iou到redis，读取文件，发送到服务器
def process_files(stop_event, HTTP_SERVER):
    # logging.getLogger().handlers.clear()
    # logger = logging.getLogger(f"process_files")
    # logger.propagate = False
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter(f"%(asctime)s - %(levelname)s %(message)s")
    # handler = logging.StreamHandler()
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    rcli = RedisDetectionDB()
    uploader = AsyncUploader()  # 初始化后台异步上传器

    while not stop_event.is_set():
        try:
            result_id, data = rcli.get_random_detection_result()
            if not result_id:
                time.sleep(1)
                continue

            logger.debug(f"Processing file: {data}")
            result_text = data["result_text"]
            image_path = data["save_path"]
            frame_num = data["frame_num"]

            file_content = read_file(image_path)
            if file_content is None or result_text == "":
                rcli.delete_record(result_id)
                time.sleep(0.01)
                continue

            last_box = rcli.redis_client.get("last_box")
            is_send = False
            result_json = json.loads(result_text)

            if last_box:
                last_box = json.loads(last_box)
                box_data = result_json["box_data"]
                is_send = calculate_iou(last_box, box_data)
            else:
                is_send = True

            if not is_send:
                delete_file(image_path)
                rcli.delete_record(result_id)
                time.sleep(0.01)
                continue

            bind_id = rcli.redis_client.get("bind_id")
            rcli.redis_client.set("last_box", json.dumps(result_json["box_data"]))

            if bind_id:
                files = {
                    "detectionDatas": (None, result_text, "application/json"),
                    "file": ("frame.jpg", bytes(file_content), "image/jpeg"),
                }

                url = f"{HTTP_SERVER}/devices/{bind_id}/detection"
                uploader.upload(
                    upload_file(url, files, frame_num, rcli)
                )  # 异步非阻塞上传
            delete_file(image_path)
            rcli.delete_record(result_id)
            time.sleep(0.01)

        except Exception as e:
            logger.error(f"Error processing files: {e}")
    loop.close()
    logger.debug("Stopping file processing loop.")
