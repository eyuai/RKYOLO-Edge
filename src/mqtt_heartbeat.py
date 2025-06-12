import psutil, datetime, time, json
import redis
import ast
from utils import get_machine_id, get_local_ip, logger

machine_id = get_machine_id()


def analyze_upload_results(redis_client: redis.Redis):
    """分析 Redis 队列中上传的结果"""
    # 获取整个队列数据
    queue_data = redis_client.lrange("upload_results", 0, -1)
    redis_client.ltrim("upload_results", 1, 0)
    yolo_infer_get_frame = redis_client.getset("yolo_infer_get_frame", 0) or "0"

    status_minus_1_count = 0  # 统计 status=-1 的数量
    success_count = 0
    frame_nums_with_status_0 = []  # 存储 status=0 的 frame_num

    for item in queue_data:
        try:
            data = ast.literal_eval(item)  # 解析字符串为字典
            if data.get("status") == -1:
                status_minus_1_count += 1
            elif data.get("status") == 0:
                frame_nums_with_status_0.append(data.get("frame_num"))
            elif data.get("status") == 1:
                success_count += 1
        except (SyntaxError, ValueError):
            print(f"解析失败: {item}")

    queue_length = len(queue_data)  # 队列总长度

    return {
        "not_box_count": status_minus_1_count,
        "fail_list": frame_nums_with_status_0,
        "success_count": success_count,
        "total": queue_length,
        "yolo_infer_get_frame": int(yolo_infer_get_frame) or 0,
    }


def get_system_info():
    # 获取内存信息
    memory_info = psutil.virtual_memory()
    memory_usage = {
        "total": memory_info.total,
        "available": memory_info.available,
        "used": memory_info.used,
        "percent": memory_info.percent,
    }

    # 获取磁盘分区信息
    disk_partitions = psutil.disk_partitions()
    disk_usage = {}
    for partition in disk_partitions:
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_usage[partition.device] = {
                "mountpoint": partition.mountpoint,
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": usage.percent,
            }
        except PermissionError:
            # 忽略没有权限访问的分区
            continue

    # 获取开机时间
    boot_time = psutil.boot_time()
    boot_time_formatted = datetime.datetime.fromtimestamp(boot_time).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # 获取CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)

    # 获取网络统计信息
    net_io = psutil.net_io_counters()
    net_io_dict = {
        "bytes_sent": net_io.bytes_sent,
        "bytes_recv": net_io.bytes_recv,
        "packets_sent": net_io.packets_sent,
        "packets_recv": net_io.packets_recv,
        "errin": net_io.errin,
        "errout": net_io.errout,
        "dropin": net_io.dropin,
        "dropout": net_io.dropout,
    }

    # 获取系统负载
    load_avg = psutil.getloadavg()

    return {
        "memory": memory_usage,
        "disk": disk_usage,
        "boot_time": boot_time_formatted,
        "cpu_percent": cpu_percent,
        "load_avg": load_avg,
        "net_io": net_io_dict,
    }


# 定时任务函数
def mqtt_heartbeat(thread_event, redis_client, mqtt_client):
    try:
        local_ip = get_local_ip()
        logger.info(f"thread_event.is_set():{thread_event.is_set()}")
        while not thread_event.is_set():  # 控制任务是否继续执行
            upload_status = analyze_upload_results(redis_client)
            system_info = get_system_info()
            heartbeat = 30
            connect_message = json.dumps(
                {
                    "local_ip": local_ip,
                    "machine_id": machine_id,
                    "upload_status": upload_status,
                    "system_info": system_info,
                    "heartbeat": heartbeat,
                }
            )
            mqtt_client.publish(f"/heartbeat/{machine_id}", connect_message)

            # 等待5秒钟，避免创建过多定时任务
            time.sleep(heartbeat)
            # gc.collect()  # 手动触发垃圾回收
    except KeyboardInterrupt:
        logger.error("程序中断，mqtt_heartbeat 正在退出...")
    except Exception as e:
        logger.error(f"Thread error: {e}")
