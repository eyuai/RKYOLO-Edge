from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)
import multiprocessing as mp
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import subprocess
from typing import List
from pathlib import Path
import paho.mqtt.client as mqtt
import argparse, uvicorn, json, os, hashlib, uuid, zipfile, threading, cv2, time, asyncio, signal, requests, shutil, base64, queue
import traceback
from multiprocessing.synchronize import Event
import gc
import redis
from linuxpy.video.device import Device
import numpy as np

from const import (
    MQTT_BROKER_ADDRESS,
    MQTT_BROKER_PORT,
    MODEL_DIR,
    DB_PATH,
    DATA_PATH,
    MQTT_TRANSPORT,
    MQTT_CLIENT_ID,
    MQTT_CLIENT_PASSWD,
    HTTP_SERVER,
    OBJ_THRESH,
    NMS_THRESH,
    SAVE_FILE_PATH,
    USE_CAMERA,
)
from src.uvc import camera_init_v4l2
from src.hik import CameraModule, get_current_frame
from src.db import (
    get_db,
    initialize_database,
    get_models,
    add_model,
    delete_model,
    find_model,
    change_model,
    find_one_model,
)

from src.detector import get_push
from src.yolo import infer_frames, init_yolo
from src.yolo_utils import draw2
from src.utils import get_machine_id, get_local_ip, logger, is_decoded_image
from src.file import process_files
from src.mqtt_heartbeat import mqtt_heartbeat, get_system_info

mp.set_start_method("fork")  # 确保 Windows 和 Linux 兼容
stop_event: Event = mp.Event()
copy_stop_event: Event = mp.Event()
thread_event = threading.Event()
get_push_thread_event = threading.Event()
asyncio_event = asyncio.Event()


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file")
args = parser.parse_args()


def load_config(config_path):
    # 加载并执行外部的 Python 配置文件
    with open(config_path, "r") as f:
        exec(f.read(), globals())  # 执行外部 Python 文件，更新当前的全局命名空间


# 加载并覆盖配置
load_config(args.config)

data_path = Path(DATA_PATH)
if not data_path.exists():
    data_path.mkdir(parents=True)  # 如果父目录不存在，自动创建
    logger.debug(f"文件夹 '{data_path}' 已创建")
else:
    logger.debug(f"文件夹 '{data_path}' 已存在")

folder_path = Path(MODEL_DIR)
if not folder_path.exists():
    folder_path.mkdir(parents=True)  # 如果父目录不存在，自动创建
    logger.debug(f"文件夹 '{folder_path}' 已创建")
else:
    logger.debug(f"文件夹 '{folder_path}' 已存在")

conn = initialize_database(DB_PATH)
conn.close()

app = FastAPI()
CONFIG_FILE_NAME = "config.json"


# 添加 CORS 中间件，允许所有来源的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)


# 获取本机特征码
machine_id = get_machine_id()


# MQTT回调函数
def on_connect(client, userdata, flags, reason_code, properties):
    logger.info(f"Connected with result code {reason_code}")
    topics = [
        f"/request/{machine_id}/ip",
        f"/request/{machine_id}/bind_id/#",
        f"/request/{machine_id}/unbind",
        f"/request/{machine_id}/update/models",
        f"/request/{machine_id}/use/model_id/#",
        f"/request/{machine_id}/use/model/list",
        f"/request/{machine_id}/del/model/model_id/#",
        f"/request/{machine_id}/reboot",
        f"/request/{machine_id}/error",
        f"/request/{machine_id}/enable_run",
        f"/request/{machine_id}/disable_run",
        f"/request/{machine_id}/system_info",
        f"/request/{machine_id}/sync_info",
        "/heartbeat",
    ]
    for topic in topics:
        client.subscribe(topic)

    local_ip = get_local_ip()
    connect_message = json.dumps(
        {"status": "ok", "code": 0, "local_ip": local_ip, "machine_id": machine_id}
    )
    logger.info(f"connect_message{connect_message}")
    try:
        with redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        ) as redis_client:
            enable_run = redis_client.get("enable_run")
        if enable_run:
            global SAVE_FILE_PATH, DB_PATH, HTTP_SERVER, OBJ_THRESH, NMS_THRESH, CONFIG_FILE_NAME
            create_infer_processes(
                SAVE_FILE_PATH,
                DB_PATH,
                HTTP_SERVER,
                OBJ_THRESH,
                NMS_THRESH,
                CONFIG_FILE_NAME,
                machine_id,
                3,
            )
    except Exception as e:
        logger.error(f"xxxx:{e}")
        pass
    client.publish(f"/response/{machine_id}/connect", connect_message)


def on_message(client, userdata, msg):

    try:
        payload = msg.payload.decode()
        topic = msg.topic
    except Exception as e:
        logger.error(f"Error decoding message: {e}")
        return
    local_ip = get_local_ip()
    topic_parts = topic.split("/")
    logger.info(f"topic: {topic}")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        client.publish(
            f"/response/{machine_id}/error",
            json.dumps(
                {"status": "error", "code": 10005, "msg": "Invalid JSON payload"}
            ),
        )
        return

    if topic == f"/request/{machine_id}/ip":
        client.publish(
            f"/response/{machine_id}/ip",
            json.dumps(
                {
                    "status": "ok",
                    "code": 0,
                    "local_ip": local_ip,
                    "machine_id": machine_id,
                }
            ),
        )
    elif topic == f"/request/{machine_id}/sync_info":
        enable_run = data.get("enable_run", 0)
        bind_id = data.get("bind_id", None)
        # with get_db(DB_PATH) as db:
        with redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        ) as redis_client:
            is_change = False
            old_bind_id = redis_client.get("bind_id")
            old_enable_run = redis_client.get("enable_run")
            # logger.warning(f"enable_run:{enable_run},bind_id:{bind_id}")
            if int(enable_run) == 1:
                if old_enable_run is None:
                    is_change = True
                redis_client.set("enable_run", 1)
            else:
                if old_enable_run and int(old_enable_run) == 1:
                    is_change = True
                redis_client.delete("enable_run")
            if bind_id:
                if (old_bind_id != bind_id) or not old_bind_id:
                    is_change = True
                redis_client.set("bind_id", bind_id)
            else:
                if old_bind_id:
                    is_change = True
                redis_client.delete("bind_id")

        client.publish(
            f"/response/{machine_id}/sync_info",
            json.dumps({"status": "ok", "code": 0}),
        )
        if is_change:
            os.kill(os.getpid(), signal.SIGTERM)
        return
    elif len(topic_parts) > 4 and topic_parts[:4] == [
        "",
        "request",
        machine_id,
        "bind_id",
    ]:
        bind_id = topic_parts[4]
        with redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        ) as redis_client:
            enable_run = redis_client.get("enable_run")
            redis_client.set("bind_id", bind_id)

        client.publish(
            f"/response/{machine_id}/bind_id/{bind_id}",
            json.dumps(
                {
                    "status": "ok",
                    "code": 0,
                }
            ),
        )
        if enable_run:
            logger.info(f"enable_run: {enable_run}")
            os.kill(os.getpid(), signal.SIGTERM)
    elif topic == f"/request/{machine_id}/system_info":
        system_info = get_system_info()
        client.publish(
            f"/response/{machine_id}/system_info",
            json.dumps(
                {
                    "status": "ok",
                    "code": 0,
                    "system_info": system_info,
                }
            ),
        )
    elif topic == f"/request/{machine_id}/unbind":
        with redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        ) as redis_client:
            redis_client.delete("bind_id")

        client.publish(
            f"/response/{machine_id}/unbind",
            json.dumps(
                {
                    "status": "ok",
                    "code": 0,
                }
            ),
        )
        os.kill(os.getpid(), signal.SIGTERM)
    elif topic == f"/request/{machine_id}/enable_run":
        with redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        ) as redis_client:
            redis_client.set("enable_run", 1)
        client.publish(
            f"/response/{machine_id}/enable_run",
            json.dumps(
                {
                    "status": "ok",
                    "code": 0,
                }
            ),
        )
        os.kill(os.getpid(), signal.SIGTERM)
    elif topic == f"/request/{machine_id}/disable_run":
        with redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        ) as redis_client:
            redis_client.delete("enable_run")

        client.publish(
            f"/response/{machine_id}/disable_run",
            json.dumps(
                {
                    "status": "ok",
                    "code": 0,
                }
            ),
        )
        os.kill(os.getpid(), signal.SIGTERM)

    elif topic == f"/request/{machine_id}/update/models":
        download_url = data.get("url")
        model_id = data.get("model_id")

        if not model_id:
            client.publish(
                f"/response/{machine_id}/update/models",
                json.dumps(
                    {"status": "error", "code": 10002, "msg": "Missing model_id"}
                ),
            )
            return

        if check_model_exists(model_id):
            client.publish(
                f"/response/{machine_id}/update/models",
                json.dumps(
                    {
                        "status": "error",
                        "code": 10003,
                        "msg": "Model already exists",
                        "model_id": model_id,
                    }
                ),
            )
            return

        if not download_url:
            client.publish(
                f"/response/{machine_id}/update/models",
                json.dumps(
                    {
                        "status": "error",
                        "code": 10004,
                        "msg": "No download URL provided",
                    }
                ),
            )
            return

        ret = handle_model_update(download_url, model_id)
        if ret == 2:
            client.publish(
                f"/response/{machine_id}/update/models",
                json.dumps(
                    {
                        "status": "error",
                        "code": 10006,
                        "msg": "Model update failed",
                    }
                ),
            )
            return
        client.publish(
            f"/response/{machine_id}/update/models",
            json.dumps({"status": "ok", "code": 0}),
        )

    elif topic == f"/request/{machine_id}/reboot":
        client.publish(
            f"/response/{machine_id}/rebooting",
            json.dumps({"msg": "rebooting", "code": 0, "status": "ok"}),
        )
        logger.info("rebooting ")

        os.kill(os.getpid(), signal.SIGTERM)

    elif len(topic_parts) > 5 and topic_parts[:5] == [
        "",
        "request",
        machine_id,
        "use",
        "model_id",
    ]:
        model_id = topic_parts[5]
        with get_db(DB_PATH) as db:
            change_model(db, model_id)
        with redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        ) as redis_client:
            enable_run = redis_client.get("enable_run")
        client.publish(
            f"/response/{machine_id}/use/model_id/{model_id}",
            json.dumps({"status": "ok", "code": 0}),
        )
        if enable_run:
            os.kill(os.getpid(), signal.SIGTERM)
        return
    elif len(topic_parts) > 6 and topic_parts[:6] == [
        "",
        "request",
        machine_id,
        "del",
        "model",
        "model_id",
    ]:
        model_id = topic_parts[6]
        with get_db(DB_PATH) as db:
            has_model = find_model(db, model_id)

        if not has_model:
            client.publish(
                f"/response/{machine_id}/del/model/model_id/{model_id}",
                json.dumps(
                    {"status": "error", "code": 10000, "msg": "Model not found"}
                ),
            )
            return

        file_path, zip_file_path, is_enable = has_model[7], has_model[5], has_model[8]
        logger.info(has_model)

        if is_enable == 1:
            client.publish(
                f"/response/{machine_id}/del/model/model_id/{model_id}",
                json.dumps(
                    {
                        "status": "error",
                        "code": 10001,
                        "msg": "Cannot delete an active model",
                    }
                ),
            )
            return

        if os.path.exists(file_path):
            shutil.rmtree(file_path)
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
        with get_db(DB_PATH) as db:
            delete_model(db, model_id)
        client.publish(
            f"/response/{machine_id}/del/model/model_id/{model_id}",
            json.dumps({"status": "ok", "code": 0}),
        )
        return

    elif topic == f"/request/{machine_id}/use/model/list":
        with get_db(DB_PATH) as db:
            models = get_models(db)
        data_list = [
            {
                "model_id": row[0],
                "name": row[1],
                "model_task": row[2],
                "model_type": row[3],
                "version": row[4],
                "zip_file_path": row[5],
                "sha256": row[6],
                "dirname": row[7],
                "enable": row[8],
                "created_at": row[9],
            }
            for row in models
        ]
        client.publish(
            f"/response/{machine_id}/use/model/list",
            json.dumps({"status": "ok", "code": 0, "data": data_list}),
        )
        return


def mqtt_init(MQTT_CLIENT_ID, MQTT_TRANSPORT):
    client_id = MQTT_CLIENT_ID

    # 初始化MQTT客户端
    mqtt_client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        transport=MQTT_TRANSPORT,
        clean_session=True,
    )
    # 设置用户名和密码
    mqtt_client.username_pw_set(username=client_id, password=MQTT_CLIENT_PASSWD)

    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    mqtt_client.connect(MQTT_BROKER_ADDRESS, MQTT_BROKER_PORT, 60)
    mqtt_client.loop_start()
    return mqtt_client


def check_model_exists(model_id):
    """检查数据库中是否已经存在该模型"""
    with get_db(DB_PATH) as db:
        query = "SELECT COUNT(*) FROM models WHERE model_id = ?"
        result = db.execute(query, (model_id,)).fetchone()
    return result[0] > 0  # 如果 COUNT > 0，说明模型已存在


def download_file(url: str, save_path: str):
    """下载文件"""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(
            f"Failed to download {url}, status code: {response.status_code}"
        )

    with open(save_path, "wb") as file:
        for chunk in response.iter_content(1024):
            file.write(chunk)

    return save_path


def process_model_zip(zip_file_path: str, model_id) -> dict:
    """
    处理模型 ZIP 文件：
    1. 解压 ZIP 文件
    2. 校验 config.json
    3. 计算 SHA256 校验值
    4. 存入数据库

    :param zip_file_path: ZIP 文件的完整路径
    :return: 处理结果
    :raises HTTPException: 处理失败时抛出异常
    """

    # 解压 ZIP 文件
    try:
        # 生成随机目录名
        dirname = f"{uuid.uuid4().hex}"
        unzip_dir = os.path.join(MODEL_DIR, dirname)
        os.makedirs(unzip_dir, exist_ok=True)
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)
    except zipfile.BadZipFile:
        logger.error("解压 ZIP 文件 error")
        os.remove(zip_file_path)
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid ZIP archive"
        )
    except Exception as e:
        logger.error(f"解压 ZIP 文件  failed: {str(e)}")

    # 检查 config.json 是否存在
    config_path = os.path.join(unzip_dir, CONFIG_FILE_NAME)
    if not os.path.exists(config_path):
        os.remove(zip_file_path)
        shutil.rmtree(unzip_dir)
        raise HTTPException(
            status_code=400, detail="config.json not found in the archive"
        )

    # 解析 config.json
    try:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
    except json.JSONDecodeError:
        logger.error("解析 config.json error")

        os.remove(zip_file_path)
        shutil.rmtree(unzip_dir)
        raise HTTPException(status_code=400, detail="config.json is not valid JSON")

    # 校验 config.json 必填字段
    required_keys = [
        "name",
        "model_type",
        "model_task",
        "version",
        "sha256",
        "classes",
    ]
    for key in required_keys:
        if key not in config:
            os.remove(zip_file_path)
            shutil.rmtree(unzip_dir)
            raise HTTPException(
                status_code=400, detail=f"Missing required field in config.json: {key}"
            )

    name = config["name"]
    model_type = config["model_type"]
    version = config["version"]
    model_task = config["model_task"]
    sha256_expected = config["sha256"]
    # img_size = config["img_size"]
    if model_type == "yolov5":
        if "anchors" not in config:
            os.remove(zip_file_path)
            shutil.rmtree(unzip_dir)
            logger.error("anchors error")

            raise HTTPException(
                status_code=400, detail=f"Missing required field in config.json: {key}"
            )

    # * 检查model_id是否存在于数据库，存在则不允许更新
    with get_db(DB_PATH) as db:
        has_model = find_model(conn=db, model_id=model_id)
    if has_model is not None:
        logger.error("has_model error")

        os.remove(zip_file_path)
        shutil.rmtree(unzip_dir)
        raise HTTPException(status_code=400, detail=f"model {model_id} is exits")

    # 获取模型文件路径
    file_in_zip_path = os.path.join(unzip_dir, name)
    if not os.path.exists(file_in_zip_path):
        os.remove(zip_file_path)
        shutil.rmtree(unzip_dir)
        raise HTTPException(
            status_code=400, detail=f"{name} file not found in the archive"
        )

    # 计算 SHA256
    sha256_calculated = hashlib.sha256()
    with open(file_in_zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_calculated.update(chunk)
    sha256_hash = sha256_calculated.hexdigest()

    if sha256_expected != sha256_hash:
        logger.error("sha256_expected error")

        os.remove(zip_file_path)
        shutil.rmtree(unzip_dir)
        raise HTTPException(
            status_code=400,
            detail="SHA256 mismatch between config.json and actual file",
        )
    with get_db(DB_PATH) as db:
        # 保存到数据库
        add_model(
            db,
            model_id,
            name,
            model_task,
            model_type,
            version,
            zip_file_path=zip_file_path,
            sha256=sha256_hash,
            dirname=os.path.join(MODEL_DIR, dirname),
            enable=0,
        )
        logger.debug("保存完成了")

        change_model(db, model_id)

    return model_id


def handle_model_update(download_url: str, model_id):
    """处理模型更新：下载资源 -> 解压 -> 校验 -> 更新数据库 -> 重启服务"""
    try:
        zip_file_path = os.path.join(MODEL_DIR, f"{uuid.uuid4().hex}.zip")

        # 下载文件
        logger.info(f"Downloading model from {download_url} to {zip_file_path}")
        download_file(download_url, zip_file_path)
        # 处理 ZIP
        result = process_model_zip(zip_file_path, model_id)
        logger.info(result)

        # 重启服务
        logger.info("Restarting service...")

        return 0
    except Exception as e:
        logger.error(f"Model update failed: {str(e)}")
        return 2


# 主线程继续执行其他任务
# 获取模型
@app.get("/models", response_model=List[dict])
async def get_models_endpoint():
    with get_db(DB_PATH) as db:
        models = get_models(db) or []
    return [
        {
            "model_id": row[0],  # 模型id
            "name": row[1],  # 模型名称
            "model_task": row[2],  # 模型类型
            "model_type": row[3],  # 模型类型
            "version": row[4],  # 版本
            "zip_file_path": row[5],  # 模型zip文件路径
            "sha256": row[6],  # sha256
            "dirname": row[7],  # 文件夹
            "enable": row[8],  # 是否启动
            "created_at": row[9],  # 创建时间
        }
        for row in models
    ]


# 添加模型
@app.post("/models", response_model=dict)
async def add_model_endpoint(
    model_id: str,
    file: UploadFile = File(...),
):
    file_extension = os.path.splitext(file.filename)[1]
    if file_extension != ".zip":
        raise HTTPException(
            status_code=400, detail="Uploaded file must be a ZIP archive"
        )

    # 保存上传的 ZIP 文件
    zip_file_path = os.path.join(MODEL_DIR, f"{uuid.uuid4().hex}{file_extension}")
    with open(zip_file_path, "wb") as buffer:
        buffer.write(await file.read())

    process_model_zip(zip_file_path, model_id)
    return {"message": "Model added successfully"}


# 更换模型，并重启服务
@app.post("/models/{model_id}/activate", response_model=dict)
async def activate_model_endpoint(model_id: str):
    with get_db(DB_PATH) as db:
        # 更换模型
        change_model(db, model_id)
    # 重启服务
    os.kill(os.getpid(), signal.SIGTERM)
    return {"message": "Model activated successfully"}


# 重启服务
@app.post("/models/{model_id}/reboot")
async def activate_model_endpoint():
    os.kill(os.getpid(), signal.SIGTERM)
    # 重启服务
    return {"message": "OK"}


# 删除模型
@app.delete("/models/{model_id}", response_model=dict)
async def delete_model_endpoint(model_id: str):
    logger.info("DB_PATH: {DB_PATH}")
    with get_db(DB_PATH) as db:
        # 获取模型信息
        model = find_model(db, model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # 删除本地文件
    file_path = model[7]
    zip_file_path = model[5]
    is_enable = model[8]
    # 判断模型是否启用
    if is_enable == 1:
        raise HTTPException(status_code=400, detail="Cannot delete an active model")
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    if os.path.exists(zip_file_path):
        os.remove(zip_file_path)

    # 删除数据库中的模型记录
    with get_db(DB_PATH) as db:
        delete_model(db, model_id)
    return {"message": "Model deleted successfully"}


nHeight = 0
nWidth = 0


def camera_init_sdk():
    """相机初始化"""
    camera_module = CameraModule()
    global nHeight, nWidth

    try:
        camera_module.initialize_sdk()
        camera_module.enum_devices()
        camera_module.connect_device()
        camera_module.start_grabbing()
        nHeight = camera_module.nHeight
        nWidth = camera_module.nWidth
        logger.info(f"Camera initialized: {nHeight}x{nWidth}")
    except Exception as e:
        logger.error(f"Camera initialization failed: {e}")
        try:
            camera_module.release()
        except Exception as e:
            logger.error(f"Camera release failed: {e}")
        return None, None, None

    return camera_module, nWidth, nHeight


def capture_frames(
    frame_queue: queue.Queue, infer_queue: queue.Queue, thread_event, USE_CAMERA
):
    """摄像头取帧进程"""
    logger.info("Frame Capture Process Started")
    global nWidth, nHeight
    # 尝试初始化摄像头
    camera_module = None
    cap = None
    iterator = None
    camera_queue = queue.Queue(maxsize=10)
    logger.warning(f"USE_CAMERA: {USE_CAMERA}")

    while True:
        if USE_CAMERA == "hik":
            camera_module, nHeight, nWidth = camera_init_sdk()
            if camera_module is not None:
                logger.info("Camera initialized successfully using SDK")
                threading.Thread(
                    target=get_current_frame,
                    args=(camera_module.cam, camera_queue, thread_event),
                    daemon=True,
                ).start()
                break
        elif USE_CAMERA == "uvc":
            device_path, nWidth, nHeight = camera_init_v4l2()
            device_id = int(device_path.replace("/dev/video", ""))
            # cap = Device.from_id(device_id)
            # iterator = iter(cap)
            if device_path is not None:
                logger.info("Camera initialized successfully using uvc")
                break
            time.sleep(0.2)

    frame_num = 0
    get_time = 0

    # 新增统计FPS用变量
    start_time = time.time()
    frames_in_one_sec = 0

    while not thread_event.is_set():
        try:
            if USE_CAMERA != "uvc":
                # 你原来的代码逻辑...
                pass

            elif USE_CAMERA == "uvc":
                # 你的UVC循环逻辑开始
                device_path, nWidth, nHeight = camera_init_v4l2()
                device_id = int(device_path.replace("/dev/video", ""))

                try:
                    with Device.from_id(device_id) as dev:
                        iterator = iter(dev)
                        logger.info("Camera initialized successfully using uvc")

                        while not thread_event.is_set():
                            try:
                                frame = next(iterator)

                                if frame is None:
                                    get_time += 1
                                    if get_time > 10:
                                        logger.error("Camera read error, retrying...")
                                        get_time = 0
                                    logger.warning(
                                        "Failed to read frame from uvc camera"
                                    )
                                    time.sleep(0.1)
                                    continue

                                frame_num += 1
                                frames_in_one_sec += 1  # 计数一秒内帧数
                                now = time.time()

                                # 每秒打印一次FPS
                                if now - start_time >= 1.0:
                                    logger.warning(
                                        f"UVC摄像头取帧FPS: {frames_in_one_sec}"
                                    )
                                    frames_in_one_sec = 0
                                    start_time = now

                                frame_data = {
                                    "frame_num": frame.frame_nb,
                                }
                                # image_np = cv2.imdecode(
                                #     np.frombuffer(frame.data, dtype=np.uint8),
                                #     cv2.IMREAD_COLOR,
                                # )
                                image_np = np.frombuffer(frame.data, dtype=np.uint8)
                                result = (0, frame_data, image_np)

                                # 推理队列
                                try:
                                    if infer_queue.full():
                                        infer_queue.get_nowait()
                                    infer_queue.put_nowait(result)
                                except queue.Full:
                                    pass

                                # 显示队列
                                try:
                                    if frame_queue.full():
                                        frame_queue.get_nowait()
                                    frame_queue.put_nowait(result)
                                except queue.Full:
                                    pass

                                time.sleep(0.004)

                            except StopIteration:
                                logger.warning("No more frames from uvc")
                                break
                            except Exception as e:
                                logger.error(f"Exception while reading frame: {e}")
                                logger.error(traceback.format_exc())
                                time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error initializing UVC device: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(1)

        except queue.Full:
            pass

        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in capture_frames: {e}")
            time.sleep(0.1)

    try:
        if cap is not None:
            if hasattr(cap, "close"):
                cap.close()
            elif hasattr(cap, "release"):
                cap.release()
        elif camera_module is not None:
            camera_module.release()
    finally:
        logger.info("Frame Capture Process Stopped")


# 共享变量
latest_frame = None
latest_frame_base64 = None
lock = threading.Lock()


def stream_frames(stop_event):
    """FFmpeg 推流进程，失败时自动重启"""
    logger.info("Frame Streaming Process Started")

    def start_ffmpeg(nWidth=nWidth, nHeight=nHeight):
        """启动 FFmpeg 进程"""
        logger.info(f"Starting FFmpeg with resolution {nWidth}x{nHeight}")
        cmd = [
            "ffmpeg",
            "-y",
            "-an",  # 无音频
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{nWidth}x{nHeight}",  # 分辨率
            "-pix_fmt",
            "yuv444p",
            "-r",
            "20",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-f",
            "flv",
            "-rtbufsize",
            "100M",
            "rtmp://127.0.0.1:1935/live/stream",  # 输出目标
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    ffmpeg_proc = None

    while not stop_event.is_set():
        try:
            if ffmpeg_proc is None or ffmpeg_proc.poll() is not None:
                # 启动或重启 FFmpeg 进程
                ffmpeg_proc = start_ffmpeg(nWidth=nWidth, nHeight=nHeight)
                logger.info("FFmpeg process started with PID %d", ffmpeg_proc.pid)

            if (
                latest_frame is not None
                and isinstance(latest_frame, bytes)
                and len(latest_frame) > 0
            ):
                logger.debug("Writing frame to FFmpeg stdin...")
                ffmpeg_proc.stdin.write(latest_frame)
                ffmpeg_proc.stdin.flush()
                logger.debug("Frame written successfully.")
            else:
                logger.warning("Invalid frame data, skipping...")

        except IOError as e:
            # 处理与 I/O 相关的错误
            logger.error(f"I/O Error: {e}")
            ffmpeg_proc = start_ffmpeg(nWidth=nWidth, nHeight=nHeight)
        except Exception as e:
            # 其他未知错误，记录日志但不重启
            logger.error(f"Unexpected error: {e}")

        time.sleep(0.02)

    # 退出时关闭 FFmpeg
    if ffmpeg_proc and ffmpeg_proc.poll() is None:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    logger.info("Frame Streaming Process Stopped")


def get_yolo():
    try:
        with get_db(DB_PATH) as db:
            model = find_one_model(db)

        unzip_dir = model[7]
        config_path = os.path.join(unzip_dir, CONFIG_FILE_NAME)
        with open(config_path, "r") as config_file:
            try:
                config = json.load(config_file)
            except json.JSONDecodeError:
                logger.error("Failed to parse config file.")
        IMG_SIZE = config.get("img_size", [640, 640])

        name = model[1]
        classes = config.get("classes", [])
        model_type = config.get("model_type", [])
        IMG_SIZE = config.get("img_size", [640, 640])
        RKNN_MODEL_PATH = os.path.join(unzip_dir, name)
        YOLO_config = {
            "OBJ_THRESH": OBJ_THRESH,
            "NMS_THRESH": NMS_THRESH,
            "IMG_SIZE": IMG_SIZE,
        }
        yolo = init_yolo(
            model_type,
            RKNN_MODEL_PATH,
            YOLO_config,
            classes,
            2,
            config,
        )
        return yolo, classes
    except Exception as e:
        logger.error(f"Failed to initialize YOLO model: {e}")
        logger.error(traceback.format_exc())
        return None, []


def copy_frame(
    frame_queue: queue.Queue, thread_event: threading.Event, redis_client: redis.Redis
):
    global latest_frame, latest_frame_base64
    class_color = [
        "red",
        "green",
        "blue",
        "yellow",
        "purple",
        "orange",
        "pink",
        "brown",
        "gray",
    ]

    times = 0
    time_at = time.time()
    enable_run = redis_client.get("enable_run")
    yolo = None
    classes = None
    boxes = None
    if enable_run is None:
        yolo, classes = get_yolo()
        color_map = dict(zip(classes, class_color))
    boxes = None
    scores = None
    real_classes = None

    while not thread_event.is_set():
        try:
            # 每秒打印一次 FPS
            if time.time() - time_at >= 1:
                logger.warning(f"[取帧FPS] 每秒取帧次数: {times}")
                times = 0
                time_at = time.time()

            ret, frame_data, frame = frame_queue.get_nowait()
            if frame_data is None or frame is None:
                time.sleep(0.02)
                continue
            if not is_decoded_image(frame):
                frame = cv2.imdecode(
                    frame,
                    cv2.IMREAD_COLOR,
                )
            if yolo is not None:

                boxes, real_classes, scores, fps, letter_box_info = yolo.detect_frame(
                    frame.copy()
                )

            with lock:  # 线程安全更新全局变量
                times += 1
                if boxes is not None:
                    logger.info(f"boxes len: {len(boxes)}")
                    frame = draw2(frame, boxes, scores, real_classes, color_map)

                _, buffer = cv2.imencode(".jpg", frame)
                latest_frame = buffer.tobytes()
                latest_frame_base64 = base64.b64encode(latest_frame).decode("utf-8")

        except queue.Empty:
            # frame_queue 为空，跳过
            pass
        except Exception as e:
            logger.error(f"copy_frame 发生异常: {e}")

        time.sleep(0.02)


async def generate_sse():
    """SSE (Server-Sent Events) 视频流"""
    global latest_frame_base64
    frame_data = None
    times = 0
    time_at = time.time()
    try:
        while not asyncio_event.is_set():
            if time_at + 1 < time.time():
                logger.info(f"times: {times} {time_at}")
                times = 0
                time_at = time.time()
            if latest_frame_base64 == frame_data:
                await asyncio.sleep(0.02)
                continue
            elif latest_frame_base64:
                times += 1
                frame_data = latest_frame_base64
            if frame_data:
                yield f"data: {frame_data}\n\n"
            await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        print("SSE 任务被取消")
        return  # 确保流终止
    except Exception as e:
        print(f"SSE 发生异常: {e}")


@app.get("/video_feeds")
async def video_feeds():
    """提供 SSE (Server-Sent Events) 视频流"""
    return StreamingResponse(generate_sse(), media_type="text/event-stream")


async def stop_server(
    for_who, redis_client: redis.Redis = None, mqtt_client: mqtt = None
):
    """关闭线程 & 进程"""
    logger.info(f"[FastAPI] {for_who} 正在关闭所有线程和进程...")
    thread_event.set()
    stop_event.set()
    get_push_thread_event.set()

    # 关闭 MQTT 客户端
    try:
        if mqtt_client:
            mqtt_client.loop_stop()
    except Exception as e:
        logger.error(f"关闭 MQTT 客户端错误: {e}")

    # for thread in threading.enumerate():
    #     if thread is not threading.current_thread():
    #         thread.join(timeout=5)

    for process in mp.active_children():
        os.kill(process.pid, signal.SIGTERM)
        process.join(timeout=5)
    logger.info("[FastAPI] 进程已经停止")
    asyncio_event.set()
    redis_client.delete("upload_results")
    redis_client.delete("yolo_infer_get_frame")
    logger.info("[FastAPI] 线程和进程已关闭，服务器即将退出")
    os._exit(0)


def create_infer_processes(
    SAVE_FILE_PATH,
    DB_PATH,
    HTTP_SERVER,
    OBJ_THRESH,
    NMS_THRESH,
    CONFIG_FILE_NAME,
    machine_id,
    p_workeres=3,
    disable=False,
):
    """创建推理进程"""
    logger.info(f"create_infer_processes started")

    global stop_event, infer_queue, get_push_thread_event, mp_infer_queues
    if disable:
        return

    time.sleep(3)
    stop_event.clear()
    get_push_thread = threading.Thread(
        target=get_push,
        args=(infer_queue, mp_infer_queues, p_workeres, get_push_thread_event),
        daemon=True,
    )
    get_push_thread.start()

    # 创建并启动推理进程
    infer_processes = [
        mp.Process(
            target=infer_frames,
            args=(
                stop_event,
                mp_infer_queues[core_id],
                SAVE_FILE_PATH,
                DB_PATH,
                HTTP_SERVER,
                OBJ_THRESH,
                NMS_THRESH,
                CONFIG_FILE_NAME,
                machine_id,
                p_workeres,
                core_id,
            ),
            daemon=True,
        )
        for core_id in range(p_workeres)
    ]
    for p in infer_processes:
        p.start()

    gc.collect()


# 初始化队列
frame_queue = queue.Queue(maxsize=20)
infer_queue = queue.Queue(maxsize=40)
mp_infer_queues = [mp.Queue(maxsize=40) for _ in range(3)]  # 每个进程一个队列


async def main():
    mqtt_client = None

    try:
        # 设置多进程启动方式
        # 连接 Redis
        redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError as e:
            logger.warning(f"无法设置启动方式：{e}")

        mqtt_client = mqtt_init(MQTT_CLIENT_ID, MQTT_TRANSPORT)

        if not os.path.exists(SAVE_FILE_PATH):
            os.makedirs(SAVE_FILE_PATH)

        # 启动线程和进程
        capture_thread = threading.Thread(
            target=capture_frames,
            args=(frame_queue, infer_queue, thread_event, USE_CAMERA),
            daemon=True,
        )

        stop_event.clear()
        get_push_thread_event.clear()

        copy_frame_thread = threading.Thread(
            target=copy_frame,
            args=(
                frame_queue,
                thread_event,
                redis_client,
            ),
            daemon=True,
        )
        mqtt_heartbeat_thread = threading.Thread(
            target=mqtt_heartbeat,
            args=(thread_event, redis_client, mqtt_client),
            daemon=True,
        )
        process_files_thread = threading.Thread(
            target=process_files,
            args=(
                thread_event,
                HTTP_SERVER,
            ),
            daemon=True,
        )

        threads = [
            capture_thread,
            process_files_thread,
            mqtt_heartbeat_thread,
            copy_frame_thread,
        ]
        for t in threads:
            t.start()
        logger.debug("infer_process 启动---------")

        # 启动服务器
        uvicorn_config = uvicorn.Config(app, host="0.0.0.0", lifespan="on", port=8020)
        server = uvicorn.Server(uvicorn_config)
        # 监听系统信号 (Ctrl+C)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(
                    stop_server(sig, redis_client, mqtt_client)
                ),
            )

        await server.serve()

    except KeyboardInterrupt:
        logger.debug("程序中断，正在退出...")
        await stop_server("KeyboardInterrupt", redis_client, mqtt_client)
    except Exception as e:
        logger.error(e)
    finally:
        logger.info("tasks finally 处理中...")


if __name__ == "__main__":
    asyncio.run(main())
