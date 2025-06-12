# const
# 这个文件定义了一些常量和配置参数
# 这些参数通过加载config.py配置文件被覆盖，修改请移步config.py

OBJ_THRESH = 0.5
NMS_THRESH = 0.4
DB_PATH = ""  # 填写绝对路径
MODEL_DIR = ""  # 模型保存文件夹
DATA_PATH = ""  # 数据文件夹

# MQTT
MQTT_BROKER_ADDRESS = ""
MQTT_BROKER_PORT = 1883
MQTT_CLIENT_ID = ""
MQTT_CLIENT_PASSWD = ""
MQTT_TRANSPORT = "websockets"

HTTP_SERVER = None
USE_CAMERA = "hik"  # 'hik' or 'uvc'
SAVE_FILE_PATH = ""  # 保存图片保存路径


def load_config(config_path):
    config = {}
    with open(config_path, "r") as f:
        exec(f.read(), config)
    globals().update(config)


def override_config(new_config):
    globals().update(new_config)
