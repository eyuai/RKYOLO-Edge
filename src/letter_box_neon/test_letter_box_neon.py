import numpy as np
import letter_box_neon
import cv2

# 创建测试图像
im = cv2.imread("./1.jpg")
new_shape = (640, 640)
pad_color = (0, 0, 0)

# 调用优化后的函数
padded_im, info = letter_box_neon.letter_box(im, new_shape, pad_color)

print("Original shape:", info["origin_shape"])
print("New shape:", info["new_shape"])
print("Padding color:", info["pad_color"])

cv2.imwrite("./2.jpg", padded_im)
