import logging
import multiprocessing as mp
import threading, time, queue
import traceback
import gc


def get_push(
    infer_queue: queue.Queue,
    mp_infer_queues: list[mp.Queue],
    mp_infer_size: int,
    thread_event: threading.Event,
):
    logger = logging.getLogger("get_push")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.warning(f"get_push started")
    while not thread_event.is_set():
        try:
            start_at = time.time()
            ret, frame_data, frame = infer_queue.get(timeout=0.2)
            start_end = time.time()
            logger.debug(f"get_push use time: {start_end - start_at}")
            if frame_data is None:
                time.sleep(0.02)
                continue
            try:
                frame_num = frame_data["frame_num"]
                logger.debug(f"push:{frame_num}")
                # 创建共享内存
                # 将共享内存的名称、形状和数据类型放入 mp_infer_queue
                queue_index = frame_num % mp_infer_size
                item = (ret, frame_data, frame.copy())

                try:
                    mp_infer_queues[queue_index].put(item, block=False)
                    time.sleep(0.02)
                except queue.Full:
                    time.sleep(0.02)
            except Exception as e:
                logger.error(f"get_push error:{e}")
                logger.error(traceback.format_exc())
                time.sleep(0.02)
                # 如果发生异常，确保共享内存被释放
        except queue.Empty:
            logger.debug("Infer queue is empty, waiting for data")
            time.sleep(0.02)
            start_end = time.time()
            logger.debug(f"get_push use time: {start_end - start_at} - queue empty")
            continue
        except Exception as e:
            logger.error(f"get_push error:{e}")
            logger.error(traceback.format_exc())
            time.sleep(0.02)
    for q in mp_infer_queues:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
    gc.collect()
    logger.warning("get_push 线程退出")
