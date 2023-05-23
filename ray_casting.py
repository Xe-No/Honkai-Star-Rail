import cv2
import numpy as np
from cv_tools import *
import time


# 计算指定角度的最大延伸距离
def calculate_distance(image, cx, cy, angle, threshold):
    rad = np.deg2rad(angle)

    r = 0
    while 1:

        r += 1
        x = int(cx+ r * np.cos(rad))
        y = int(cy+ r * np.sin(rad))
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            break        
        # print([r,x,y])
        pixel = image[y,x]
        # print(pixel)
        if pixel < threshold:
            break
        

        # time.sleep(1)
    return [r, x, y]

# 计算整个图像中不同角度对应延伸距离
def ray_casting(image, threshold, step=15):
    height, width = image.shape
    center_x = width // 2
    center_y = height // 2
    max_distance = 0
    max_angle = 0
    canvas = image.copy()
    angle_r = []
    for angle in range(0, 360, step):
        distance, x, y = calculate_distance(image, center_x, center_y, angle, threshold)
        # if distance > max_distance:
        #     max_distance = distance
        #     max_angle = angle
        # print(x,y)
        # cv2.line(canvas, (center_x, center_y), (x, y), 128, 1)
        angle_r.append([angle, distance])

    # cv2.imshow('',canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return angle_r


if __name__ == '__main__':

    max_distance = 0
    max_angle = 0
    threshold = 70

    image = cv2.imread('test.png')
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    image =  cv2.dilate(image, kernel, iterations=1)
    image_b = get_binary(image, 30)
    print(image)
    max_angle, max_distance  = ray_casting(image_b, threshold=128, step=15)
    # 输出最大延伸距离和对应的角度
    print(f"最大延伸距离为 {max_distance}，对应角度为 {max_angle} 度。")

