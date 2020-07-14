import cv2
import numpy as np


def image_process(file_path):
    img = cv2.imread(file_path, 0)
    blur = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯模糊
    ret, binary = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)  # 二值化

    kernel = np.ones((10, 10), np.uint8)
    erosion = cv2.erode(binary, kernel)         # 膨胀
    dilation = cv2.dilate(erosion, kernel)      # 腐蚀

    left, top, right, bottom = 66535, 66535, 0, 0
    # 获取数字边界
    height, width = dilation.shape
    for i in range(height):
        for j in range(width):
            if binary[i, j] == 0:
                left = min(left, j)
                right = max(right, j)
                top = min(top, i)
                bottom = max(bottom, i)
    left = max(left-100, 0)
    right = min(right+100, width)
    top = max(top-100, 0)
    bottom = min(bottom+100, height)

    img = binary[top:bottom, left:right]
    img = cv2.resize(img, (28, 28))

    h0, w0 = img.shape
    pic_data = []
    for hx in range(h0):
        pic_data_element = []
        for wx in range(w0):
            pic_data_element.append(img[hx, wx])
        pic_data.append(pic_data_element)

    return pic_data
