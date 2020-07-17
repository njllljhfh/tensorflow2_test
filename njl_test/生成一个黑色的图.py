# -*- coding:utf-8 -*-

import numpy as np
import cv2

img_array = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 背景色(黑色)
cv2.imwrite("black.png", img_array)
