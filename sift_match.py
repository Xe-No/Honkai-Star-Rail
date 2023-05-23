from cv_tools import *
import cv2 as cv

map_prefix = './temp/Simulated_Universe/map_raw/'
masked_map_prefix = './temp/Simulated_Universe/map_masked/'
template_prefix = './temp/Simulated_Universe/map_template_sample/'
result_prefix = './temp/Simulated_Universe/map_result/'



def sift_match(img1, img2):
    # 创建SIFT对象
    sift = cv.SIFT_create()

    # 检测关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)


    # 建立FLANN匹配对象
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # 根据描述符进行匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # 筛选最优匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    img_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

    return img_match



