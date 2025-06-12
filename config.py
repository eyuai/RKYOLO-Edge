OBJ_THRESH = 0.5
NMS_THRESH = 0.4

DATA_PATH = ""  # 数据文件夹
DB_PATH = ""  # 数据库路径
MODEL_DIR = ""  # 模型保存文件夹
# MQTT 配置
MQTT_BROKER_ADDRESS = ""
MQTT_BROKER_PORT = 80
MQTT_CLIENT_ID = ""
MQTT_CLIENT_PASSWD = ""
MQTT_TRANSPORT = "websockets"

# HTTP 服务器配置
HTTP_SERVER = ""
# RTMP 推流地址
RTMP_URL = "rtmp://127.0.0.1:1935/live/video"
# 摄像头配置，目前支持海康摄像头和uvc
USE_CAMERA = "hik"  # 'hik' or 'uvc'
# 保存图片保存路径
SAVE_FILE_PATH = ""
