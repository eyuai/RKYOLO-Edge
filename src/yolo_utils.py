import numpy as np
import letter_box_neon
import os
import cv2
from utils import logger
from PIL import Image, ImageDraw, ImageFont


def load_config(config_path):
    config = {}
    with open(config_path, "r") as f:
        exec(f.read(), config)
    globals().update(config)


def override_config(new_config):
    globals().update(new_config)


# 使用 letter_box 对图像进行填充


# 使用 letter_box 对图像进行填充
# def letter_box(im, new_shape, pad_color=(0, 0, 0)):
#     shape = im.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

#     dw = new_shape[1] - new_unpad[0]
#     dh = new_shape[0] - new_unpad[1]

#     left, right = np.floor(dw / 2), np.ceil(dw / 2)
#     top, bottom = np.floor(dh / 2), np.ceil(dh / 2)

#     im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     im = cv2.copyMakeBorder(
#         im,
#         int(top),
#         int(bottom),
#         int(left),
#         int(right),
#         cv2.BORDER_CONSTANT,
#         value=pad_color,
#     )

#     letter_box_info = {
#         "origin_shape": tuple(shape),
#         "new_shape": tuple(new_shape),
#         "w_ratio": float(r),
#         "h_ratio": float(r),
#         "dw": int(dw),
#         "dh": int(dh),
#         "r": float(r),
#         "pad_color": tuple(map(int, pad_color)),
#     }

#     return im, letter_box_info


# neon
letter_box = letter_box_neon.letter_box
# rga neon
# letter_box = letterbox.letter_box


def yolov5_box_process(position, anchors, img_size=(640, 640)):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h), indexing="xy")
    grid = np.stack((col, row), axis=0).reshape(1, 2, grid_h, grid_w)

    stride = np.array(
        [img_size[0] // grid_h, img_size[1] // grid_w], dtype=np.float32
    ).reshape(1, 2, 1, 1)

    # # 使用 np.tile 代替 np.repeat
    # col = np.tile(col.reshape(1, 1, grid_h, grid_w), (len(anchors), 1, 1, 1))
    # row = np.tile(row.reshape(1, 1, grid_h, grid_w), (len(anchors), 1, 1, 1))
    # 使用 np.broadcast_to 代替 np.tile，避免额外的内存复制
    col = np.broadcast_to(
        col.reshape(1, 1, grid_h, grid_w), (len(anchors), 1, grid_h, grid_w)
    )
    row = np.broadcast_to(
        row.reshape(1, 1, grid_h, grid_w), (len(anchors), 1, grid_h, grid_w)
    )

    # 直接转换 anchors，避免额外的 reshape
    anchors = np.asarray(anchors, dtype=np.float32)[..., None, None]

    # 计算框的位置
    box_xy = position[:, :2] * 2 - 0.5 + grid
    box_wh = (position[:, 2:4] * 2) ** 2 * anchors

    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    # 计算 xyxy
    xyxy = np.empty_like(box)
    xyxy[:, 0] = box[:, 0] - box[:, 2] / 2
    xyxy[:, 1] = box[:, 1] - box[:, 3] / 2
    xyxy[:, 2] = box[:, 0] + box[:, 2] / 2
    xyxy[:, 3] = box[:, 1] + box[:, 3] / 2

    # del grid, stride, col, row, box_xy, box_wh, box, anchors
    # gc.collect()

    return xyxy


def yolov5_post_process(
    input_data, anchors, NMS_THRESH, OBJ_THRESH, img_size=(640, 640)
):
    boxes, scores, classes_conf = [], [], []
    # 1*255*h*w -> 3*85*h*w
    input_data = [
        _in.reshape([len(anchors[0]), -1] + list(_in.shape[-2:])) for _in in input_data
    ]
    for i in range(len(input_data)):
        boxes.append(
            yolov5_box_process(input_data[i][:, :4, :, :], anchors[i], img_size)
        )
        scores.append(input_data[i][:, 4:5, :, :])
        classes_conf.append(input_data[i][:, 5:, :, :])

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf, OBJ_THRESH)
    # nms
    nboxes, nclasses, nscores = [], [], []

    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s, NMS_THRESH)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    # del nboxes
    # del nclasses
    # del nscores
    # gc.collect()
    return boxes, classes, scores


# 使用 get_real_box 还原坐标
def get_real_box(boxes, letter_box_info, in_format="xyxy"):
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)

    if letter_box_info:
        dw, dh = letter_box_info["dw"], letter_box_info["dh"]
        r = letter_box_info["r"]  # 统一缩放比例
        origin_h, origin_w = letter_box_info["origin_shape"]

        if in_format == "xyxy":
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw / 2) / r  # x 坐标
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh / 2) / r  # y 坐标
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, origin_w)  # x 坐标裁剪
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, origin_h)  # y 坐标裁剪

    return boxes


def get_real_seg(seg, letter_box_info):
    # Fix side effect
    dh = int(letter_box_info["dh"])
    dw = int(letter_box_info["dw"])
    origin_shape = letter_box_info["origin_shape"]
    new_shape = letter_box_info["new_shape"]
    if (dh == 0) and (dw == 0) and origin_shape == new_shape:
        return seg
    elif dh == 0 and dw != 0:
        seg = seg[:, :, dw:-dw]  # a[0:-0] = []
    elif dw == 0 and dh != 0:
        seg = seg[:, dh:-dh, :]
    seg = np.where(seg, 1, 0).astype(np.uint8).transpose(1, 2, 0)
    seg = cv2.resize(
        seg, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_LINEAR
    )
    if len(seg.shape) < 3:
        return seg[None, :, :]
    else:
        return seg.transpose(2, 0, 1)


# 加载 RKNN 模型
def load_model(RKNN_MODEL_PATH, core_id):
    if not os.path.exists(RKNN_MODEL_PATH):
        # logger.error(f"模型文件不存在: {RKNN_MODEL_PATH}")
        raise RuntimeError(f"模型文件不存在: {RKNN_MODEL_PATH}")
    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    # logger.info(f"--> 加载 RKNN 模型: {RKNN_MODEL_PATH}")
    if rknn.load_rknn(RKNN_MODEL_PATH) != 0:
        # logger.info("加载 RKNN 模型失败！")
        raise RuntimeError("加载 RKNN 模型失败！")

    if core_id == 0:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif core_id == 1:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif core_id == 2:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif core_id == -1:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError("初始化运行时环境失败！")

    # logger.info("--> 初始化运行时环境")
    # if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1) != 0:
    #     # logger.info("初始化运行时环境失败！")
    #     raise RuntimeError("初始化运行时环境失败！")

    logger.info(f"RKNN 模型加载成功！{core_id}")
    return rknn


def filter_boxes(boxes, box_confidences, box_class_probs, OBJ_THRESH):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def draw(image, boxes, scores, classes, CLASSES):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        logger.info(
            "%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score)
        )
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            image,
            "{0} {1:.2f}".format(CLASSES[cl], score),
            (top, left - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )


font = None


def draw2(image, boxes, scores, classes, color_map):
    # Convert OpenCV image to PIL image
    global font
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    original_width, original_height = image_pil.size
    # logger.info(f"Image size: {original_width}x{original_height}")

    # 根据图像大小调整字体大小和线条宽度
    font_size = int(original_width / 640 * 30)
    line_width = int(original_width / 640 * 4)

    try:
        if font == None:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", font_size
            )
    except IOError:
        logger.warning("Failed to load specified font, using default font.")
        font = ImageFont.load_default()

    for box, score, cl in zip(boxes, scores, classes):
        left, top, right, bottom = map(int, box)

        # 绘制矩形框
        draw.rectangle(
            [left, top, right, bottom],
            outline=color_map.get(cl, "black"),
            width=line_width,
        )

        # 绘制文本背景
        text = f"{cl} {score:.2f}"
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_bg_left = left
        text_bg_top = max(0, top - text_height - 5)
        text_bg_right = left + text_width + 10
        text_bg_bottom = text_bg_top + text_height + 5
        draw.rectangle(
            [text_bg_left, text_bg_top, text_bg_right, text_bg_bottom],
            fill=color_map.get(cl, "black"),
        )

        # 绘制文本
        draw.text(
            (text_bg_left + 5, text_bg_top + 2),
            text,
            font=font,
            fill="black",  # 文本颜色改为白色以便更好地显示
        )

    # Convert PIL image back to OpenCV image
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return image_cv


def img_check(path):
    img_type = [".jpg", ".jpeg", ".png", ".bmp"]
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


def nms_boxes(boxes, scores, NMS_THRESH):
    """Suppress non-maximal boxes.
    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]

    return np.array(keep)


# def dfl(position):
#     # Distribution Focal Loss (DFL)

#     x = torch.tensor(position)
#     n, c, h, w = x.shape
#     p_num = 4
#     mc = c // p_num
#     y = x.reshape(n, p_num, mc, h, w)
#     y = y.softmax(2)
#     acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
#     y = (y * acc_metrix).sum(2)
#     return y.numpy()


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def dfl(position):
    x = np.asarray(position, dtype=np.float32)  # 避免额外复制
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = softmax(y, axis=2)

    acc_metrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    np.multiply(y, acc_metrix, out=y)  # 避免创建新数组
    y = np.sum(
        y, axis=2, keepdims=False
    )  # 直接使用 keepdims=False 以减少不必要的维度扩展

    return y


def box_process(position, img_size=(640, 640)):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array(
        [img_size[0] // grid_h, img_size[1] // grid_w], dtype=np.float32
    ).reshape(1, 2, 1, 1)

    if position.shape[1] == 4:
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    else:
        position = dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    del position
    return xyxy


def sp_flatten(_in):
    """扁平化数据用于后处理"""
    ch = _in.shape[1]
    return _in.transpose(0, 2, 3, 1).reshape(-1, ch)


def post_process(input_data, NMS_THRESH, OBJ_THRESH, img_size=(640, 640)):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch

    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i], img_size))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(
            np.ones_like(
                input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32
            )
        )

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf, OBJ_THRESH)
    del classes_conf

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s, NMS_THRESH)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    # del nboxes, nclasses, nscores
    # gc.collect()  # 强制释放未引用的 numpy 内存

    return boxes, classes, scores
