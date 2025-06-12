import sqlite3
from contextlib import contextmanager
from utils import logger
import uuid
import redis


# 初始化数据库
def initialize_database(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 设置 WAL 模式，提升并发性能
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")

    # 创建 models 表，并确保 model_id 唯一
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,  -- 确保 model_id 唯一
            name TEXT, -- 模型名
            model_task TEXT,  -- 模型任务类型
            model_type TEXT, -- 模型类型，如 yolov5
            version TEXT, -- 自定义版本号
            zip_file_path TEXT, -- zip文件路径
            sha256 TEXT, -- rknn文件的sha256
            dirname TEXT, -- 文件夹名
            enable INTEGER, -- 是否启动
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    return conn


# 使用上下文管理器处理数据库连接
@contextmanager
def get_db(DB_PATH):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    try:
        yield conn
    finally:
        conn.close()


class RedisDetectionDB:
    def __init__(self, host="localhost", port=6379, db=0):
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, decode_responses=True
        )

    def create_placeholder(self, save_path: str, frame_num: int) -> str:
        record_id = f"detection_result:{uuid.uuid4()}"
        self.redis_client.hset(
            record_id,
            mapping={
                "save_path": save_path,
                "frame_num": str(frame_num),
                "result_text": "",
            },
        )
        return record_id  # 返回 key

    def get_record(self, record_id: str) -> dict:
        return self.redis_client.hgetall(record_id)

    def get_random_detection_result(self):
        keys = list(self.redis_client.scan_iter("detection_result:*"))
        for key in keys:
            result_text = self.redis_client.hget(key, "result_text")
            if result_text:  # 非空字符串
                # non_empty_keys.append(key)
                return key, self.redis_client.hgetall(key)

        # if not non_empty_keys:
        return None, None

    def update_result_text(self, record_id: str, result_text: str) -> bool:
        if self.redis_client.exists(record_id):
            self.redis_client.hset(record_id, "result_text", result_text)
            return True
        return False

    def delete_record(self, record_id: str) -> bool:
        return self.redis_client.delete(record_id) > 0


# 获取所有模型
def get_models(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models")
    return cursor.fetchall()


# 根据 model_id 查找模型
def find_model(conn: sqlite3.Connection, model_id):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE model_id = ? LIMIT 1", (model_id,))
    return cursor.fetchone()


# 获取当前启用的最新模型
def find_one_model(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM models WHERE enable = 1 ORDER BY created_at DESC LIMIT 1"
    )
    return cursor.fetchone()


# 修改启用的模型
def change_model(conn: sqlite3.Connection, model_id):
    try:

        cursor = conn.cursor()
        cursor.execute("BEGIN TRANSACTION")
        cursor.execute("UPDATE models SET enable = 0 WHERE enable = 1")
        cursor.execute("UPDATE models SET enable = 1 WHERE model_id = ?", (model_id,))
        conn.commit()
    except sqlite3.OperationalError as e:
        conn.rollback()
        logger.error(f"Error: {e}")


# 添加模型
def add_model(
    conn: sqlite3.Connection,
    model_id,
    name,
    model_task,
    model_type,
    version,
    zip_file_path,
    sha256,
    dirname,
    enable,
):
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO models (model_id, name, model_task, model_type, version, zip_file_path, sha256, dirname, enable)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                name,
                model_task,
                model_type,
                version,
                zip_file_path,
                sha256,
                dirname,
                enable,
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        logger.warning(f"Error: model_id {model_id} already exists in the database.")
        return False  # model_id 已存在，插入失败


# 删除模型
def delete_model(conn: sqlite3.Connection, model_id):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
    conn.commit()


# 更新模型信息（支持部分字段更新）
def update_model(
    conn: sqlite3.Connection,
    model_id,
    name=None,
    model_task=None,
    model_type=None,
    version=None,
    zip_file_path=None,
    sha256=None,
    dirname=None,
    enable=None,
):
    cursor = conn.cursor()
    fields = []
    values = []

    if name is not None:
        fields.append("name = ?")
        values.append(name)
    if model_task is not None:
        fields.append("model_task = ?")
        values.append(model_task)
    if type is not None:
        fields.append("model_type = ?")
        values.append(model_type)
    if version is not None:
        fields.append("version = ?")
        values.append(version)
    if zip_file_path is not None:
        fields.append("zip_file_path = ?")
        values.append(zip_file_path)
    if sha256 is not None:
        fields.append("sha256 = ?")
        values.append(sha256)
    if dirname is not None:
        fields.append("dirname = ?")
        values.append(dirname)
    if enable is not None:
        fields.append("enable = ?")
        values.append(enable)

    if not fields:
        logger.warning("No fields to update.")
        return False

    values.append(model_id)
    query = f"UPDATE models SET {', '.join(fields)} WHERE model_id = ?"

    cursor.execute(query, tuple(values))
    conn.commit()
    return True


# 获取所有启用的模型
def get_enabled_models(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE enable = 1")
    return cursor.fetchall()


# 统计模型总数
def count_models(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM models")
    return cursor.fetchone()[0]


# 判断某个 model_id 是否存在
def model_exists(conn: sqlite3.Connection, model_id):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT EXISTS(SELECT 1 FROM models WHERE model_id = ?)", (model_id,)
    )
    return cursor.fetchone()[0] == 1
