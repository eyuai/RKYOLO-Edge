from src.yolo_utils import (
    np,
    post_process,
    yolov5_post_process,
    load_model,
    letter_box,
    get_real_box,
    override_config,
)
from multiprocessing.synchronize import Event
import multiprocessing as mp
from utils import get_local_ip, is_decoded_image
import cv2, json, uuid, time, asyncio, gc, queue, traceback, threading
from datetime import datetime
from src.file import upload_file, delete_file

# from utils import logger
import copy
import logging
import redis, uuid
from db import (
    RedisDetectionDB,
    get_db,
    find_one_model,
)
import os, sys

logging.getLogger("asyncio.selector_events").setLevel(logging.WARNING)
logger = logging.getLogger("yolo")
logger.propagate = False
logger.setLevel(logging.INFO)


class YOLOBase:
    def __init__(self, model_path, config, core_id):
        self.model = None
        self.anchors = None
        self.classes = None
        self.load_model(model_path, core_id)
        override_config(config)
        self.config = config

    def load_model(self, model_path, core_id):
        """Load the YOLO model from file"""
        logger.info(f"load_model core_id:{core_id}")
        self.model = load_model(model_path, core_id)

    def detect(self, image_path):
        raise NotImplementedError("Detect method must be implemented in subclass")


class YOLOv5(YOLOBase):
    def __init__(self, model_path, config, anchors, CLASSES, core_id):
        super().__init__(model_path, config, core_id)
        self.anchors = anchors
        self.classes = CLASSES

    def preprocess(self, img_src):
        pad_color = (0, 0, 0)
        img, letter_box_info = letter_box(
            im=img_src, new_shape=self.config["IMG_SIZE"], pad_color=pad_color
        )
        new_img = np.expand_dims(img, 0)
        return new_img, letter_box_info

    def postprocess(self, outputs, letter_box_info):
        if not outputs:
            return None, None, None
        try:
            boxes, classes, scores = yolov5_post_process(
                outputs,
                self.anchors,
                self.config["NMS_THRESH"],
                self.config["OBJ_THRESH"],
                self.config["IMG_SIZE"],
            )
            if boxes is not None:
                boxes = get_real_box(boxes, letter_box_info)

            return boxes, classes, scores
        except Exception as e:
            logger.exception(f"yolov5_post_process: {e}")
            return None, None, None

    def _process_image(self, img_src):
        """
        处理图像的公共部分，包括预处理、推理、后处理和有效性检查。
        返回 boxes, real_classes, scores, classes 或 None（如果无效）
        """

        # 图像预处理
        img, letter_box_info = self.preprocess(img_src)

        start_time = time.time()
        outputs = self.model.inference([img])
        end_time = time.time()
        inference_time = end_time - start_time
        fps = 1 / inference_time if inference_time > 0 else 0
        # 后处理
        boxes, classes, scores = self.postprocess(outputs, letter_box_info)
        # 如果任意结果为 None，则返回 None
        if boxes is None or classes is None or scores is None:
            return None, None, None, None, None

        # 确保 classes 是有效的索引
        if any(c >= len(self.classes) for c in classes):
            del classes
            return None, None, None, None, None

        real_classes = [self.classes[i] for i in classes]
        del classes, img

        return boxes, real_classes, scores, fps, letter_box_info

    def detect_frame(self, frame):
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
        results = self._process_image(frame)

        return results

    def detect(self, image_path):
        img_src = cv2.imread(image_path)
        if img_src is None:
            return None, None, None, None, None

        results = self._process_image(img_src)

        del img_src  # 显式释放
        return results


class YOLOv8(YOLOBase):
    def __init__(
        self,
        model_path,
        config,
        CLASSES,
        core_id,
    ):
        super().__init__(model_path, config, core_id)
        self.classes = CLASSES

    def preprocess(self, img_src):
        pad_color = (0, 0, 0)
        img, letter_box_info = letter_box(
            im=img_src, new_shape=self.config["IMG_SIZE"], pad_color=pad_color
        )
        new_img = np.expand_dims(img, 0)
        return new_img, letter_box_info

    def postprocess(self, outputs, letter_box_info):
        if not outputs:
            return None, None, None
        try:
            boxes, classes, scores = post_process(
                outputs,
                self.config["NMS_THRESH"],
                self.config["OBJ_THRESH"],
                self.config["IMG_SIZE"],
            )
            if boxes is not None:
                new_boxes = get_real_box(boxes, letter_box_info)
                del boxes  # 释放旧的 boxes
                boxes = new_boxes
            del letter_box_info, outputs
            return boxes, classes, scores
        except Exception as e:
            logger.info(f"YOLO 8 11 post_process: {e}")
            return None, None, None

    def _process_image(self, img_src):
        """
        处理图像的公共部分，包括预处理、推理、后处理和有效性检查。
        返回 boxes, real_classes, scores, classes 或 None（如果无效）
        """

        # 图像预处理
        img, letter_box_info = self.preprocess(img_src)

        start_time = time.time()
        outputs = self.model.inference([img])
        end_time = time.time()
        inference_time = end_time - start_time
        fps = 1 / inference_time if inference_time > 0 else 0
        # 后处理

        boxes, classes, scores = self.postprocess(outputs, letter_box_info)
        # 如果任意结果为 None，则返回 None
        if boxes is None or classes is None or scores is None:
            del letter_box_info, classes, img
            # gc.collect()
            return None, None, None, None, None

        # 确保 classes 是有效的索引
        if any(c >= len(self.classes) for c in classes):
            del classes, img
            return None, None, None, None, None

        real_classes = [self.classes[i] for i in classes]
        del classes, img
        return boxes, real_classes, scores, fps, letter_box_info

    def detect_frame(self, frame):
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
        results = self._process_image(frame)

        del frame  # 显式释放
        # gc.collect()
        return results

    def detect(self, image_path):
        img_src = cv2.imread(image_path)
        if img_src is None:
            return None, None, None, None, None

        results = self._process_image(img_src)

        del img_src  # 显式释放
        return results


class YOLOv11(YOLOv8):
    def __init__(self, model_path, config, CLASSES, core_id):
        super().__init__(model_path, config, CLASSES, core_id)


def encode_image(frame):
    try:
        image_copy = frame.copy()
        _, img_byte_arr = cv2.imencode(".jpg", image_copy)
        img_bytes = bytes(img_byte_arr)
        return img_bytes
    except Exception as e:
        logger.error(f"图像编码错误: {e}")
        return None


# 处理栈
def yolo_process_stack(
    frame,
    frame_num,
    yolo: YOLOv5 | YOLOv8 | YOLOv11,
    bind_id,
    HTTP_SERVER: str = None,
    local_ip: str = None,
    machine_id: str = None,
    model_id: str = None,
    SAVE_FILE_PATH: str = None,
    DB_PATH: str = None,
    rcli: RedisDetectionDB = None,
):
    # # 获取特定 logger
    logger = logging.getLogger(f"yolo_process_stack")
    # logger.propagate = False
    logger.setLevel(logging.INFO)

    if frame is None:
        logger.error("cv2.imdecode failed, frame is None")
        return

    result_id = None
    save_path = None
    try:
        boxes, real_classes, scores, fps, letter_box_info = yolo.detect_frame(
            frame.copy()
        )
        if boxes is None:
            rcli.redis_client.rpush(
                "upload_results",
                str({"frame_num": frame_num, "status": -1}),
            )
            time.sleep(0.01)
            return
        box_data = []
        for box, score, cl in zip(boxes, scores, real_classes):
            xmin, ymin, xmax, ymax = [int(_b) for _b in box]
            x, y = xmin, ymin
            w, h = max(0, xmax - xmin), max(0, ymax - ymin)
            box_data.append(
                {
                    "id": str(uuid.uuid4()),
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "label": cl,
                    "score": float(score),
                }
            )

        image_width, image_height = (
            letter_box_info["origin_shape"][1],
            letter_box_info["origin_shape"][0],
        )

        connect_message = {
            "local_ip": local_ip,
            "machine_id": machine_id,
            "box_data": box_data,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "image_width": image_width,
            "image_height": image_height,
            "fps": float(fps),
            "model_id": model_id,
            "frame_num": frame_num,
        }

        connect_message_json = json.dumps(connect_message)
        if SAVE_FILE_PATH and DB_PATH:
            try:
                file_name = uuid.uuid4()
                save_path = os.path.join(SAVE_FILE_PATH, f"{file_name}.jpg")
                cv2.imwrite(save_path, frame)
                result_id = rcli.create_placeholder(save_path, frame_num)
                rcli.update_result_text(result_id, connect_message_json)
            except Exception as e:
                logger.error(f"更新数据库错误: {e}")
                delete_file(save_path)
        else:
            image_copy = copy.deepcopy(frame)
            _, img_byte_arr = cv2.imencode(".jpg", image_copy)

            img_bytes = bytes(img_byte_arr)
            files = {
                "detectionDatas": (None, connect_message_json, "application/json"),
                "file": ("frame.jpg", img_bytes, "image/jpeg"),
            }

            if HTTP_SERVER and bind_id:

                url = f"{HTTP_SERVER}/devices/{bind_id}/detection"
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)  # 将新循环设置为当前线程的事件循环
                loop.run_until_complete(
                    upload_file(url, files, frame_num)
                )  # 等待文件上传完成
                loop.close()
        time.sleep(0.01)

    except Exception as e:
        logger.error(f"推理过程错误: {e}")
        if save_path:
            logger.error(f"删除文件: {save_path}")
            delete_file(save_path)
        return None


def init_yolo(
    model_type,
    RKNN_MODEL_PATH,
    YOLO_config,
    classes,
    core_id,
    config,
):
    """初始化 YOLO 模型"""
    logger.info(f"Initializing YOLO model on core {core_id}")
    # 这里可以添加 YOLO 模型的初始化代码
    if model_type == "yolov5":
        anchors = config.get("anchors", [])
        yolo = YOLOv5(
            RKNN_MODEL_PATH,
            config=YOLO_config,
            anchors=anchors,
            CLASSES=classes,
            core_id=core_id % 3,
        )
    elif model_type == "yolov8":
        yolo = YOLOv8(
            RKNN_MODEL_PATH,
            config=YOLO_config,
            CLASSES=classes,
            core_id=core_id % 3,
        )
    else:
        yolo = YOLOv11(
            RKNN_MODEL_PATH,
            config=YOLO_config,
            CLASSES=classes,
            core_id=core_id % 3,
        )

    return yolo


def infer_frames(
    stop_event: Event,
    mp_infer_queue: mp.Queue,
    SAVE_FILE_PATH,
    DB_PATH,
    HTTP_SERVER,
    OBJ_THRESH,
    NMS_THRESH,
    CONFIG_FILE_NAME,
    machine_id,
    p_workers=3,
    core_id=0,
):
    logging.getLogger().handlers.clear()
    logger = logging.getLogger(f"logging_core_{core_id}")
    logger.propagate = False
    logger.setLevel(logging.INFO)

    logger.warning(f"infer_frames 子进程启动 {core_id}")

    # 初始化变量
    rknn = None
    yolo: YOLOv5 | YOLOv8 | YOLOv11 = None
    bind_id = None

    # 加载模型配置
    with get_db(DB_PATH) as db:
        model = find_one_model(db)
    with redis.Redis(
        host="localhost", port=6379, db=0, decode_responses=True
    ) as redis_client:
        bind_id = redis_client.get("bind_id")
        if bind_id is None:
            return
        # bind_id = bind_id

    try:
        logger.info(f"Start BIND_ID: {bind_id}")
        if not model:
            logger.warning("No model found in the database.")
            return

        unzip_dir = model[7]
        config_path = os.path.join(unzip_dir, CONFIG_FILE_NAME)

        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return

        with open(config_path, "r") as config_file:
            try:
                config = json.load(config_file)
            except json.JSONDecodeError:
                logger.error("Failed to parse config file.")
                return

        classes = config.get("classes", [])
        model_type = config.get("model_type", [])
        model_id = config.get("model_id", [])
        IMG_SIZE = config.get("img_size", [640, 640])

        name = model[1]
        local_ip = get_local_ip()
        RKNN_MODEL_PATH = os.path.join(unzip_dir, name)
        YOLO_config = {
            "OBJ_THRESH": OBJ_THRESH,
            "NMS_THRESH": NMS_THRESH,
            "IMG_SIZE": IMG_SIZE,
        }
        http_server = HTTP_SERVER

        logger.info(f"core_id:{core_id}: use_core:{core_id % 3}")

        # 初始化 YOLO 模型
        try:
            yolo = init_yolo(
                model_type,
                RKNN_MODEL_PATH,
                YOLO_config,
                classes,
                core_id,
                config,
            )
            rknn = yolo.model
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            logger.error(traceback.format_exc())
        rcli = RedisDetectionDB(host="localhost", port=6379, db=0)
        while not stop_event.is_set():
            try:
                ret, frame_data, frame = mp_infer_queue.get_nowait()

                frame_num = frame_data["frame_num"]

                # 更新 Redis 统计
                rcli.redis_client.incr("yolo_infer_get_frame")

                # 提交线程任务
                logger.debug(
                    f"yolo infer p_workers: {p_workers},core_id: {core_id}, run: {frame_num}"
                )
                if not is_decoded_image(frame):
                    frame = cv2.imdecode(
                        frame,
                        cv2.IMREAD_COLOR,
                    )

                thread_task = threading.Thread(
                    target=yolo_process_stack,
                    args=(
                        frame,
                        frame_num,
                        yolo,
                        bind_id,
                        http_server,
                        local_ip,
                        machine_id,
                        model_id,
                        SAVE_FILE_PATH,
                        DB_PATH,
                        rcli,
                    ),
                    daemon=True,
                )
                thread_task.start()
                thread_task.join(2)
                time.sleep(0.01)

            except queue.Empty:
                logger.debug(f"Queue is empty, waiting for new frames... {core_id}")
                time.sleep(0.02)
            except Exception as e:
                logger.error(f"Inference process error: {e} {core_id}")
                logger.error(traceback.format_exc())
                time.sleep(0.02)
    finally:
        logger.error(f"开始关闭 推理进程 {core_id}")
        while not mp_infer_queue.empty():
            try:
                item = mp_infer_queue.get_nowait()
                if item[1] is None:
                    continue
                logger.info(f"mp_infer_queue: {item[1]['frame_num']}")
            except queue.Empty:
                break
        if rknn and hasattr(rknn, "release"):
            try:
                rknn.release()
            except Exception as e:
                logger.error(f"Error while releasing rknn model: {e}")
        gc.collect()
        logger.info(f"退出子进程{core_id}")
        sys.exit(0)
