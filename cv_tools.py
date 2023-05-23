#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Created by Xe-No at 2023/5/17
# 一些cv工具

from tools.switch_window import switch_window
from tools.get_angle import get_angle
from tools.calculated import calculated
import time 
import cv2 as cv
import numpy as np
import win32gui, win32api, win32con
import pyautogui


def get_binary(img, threshold=200):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    return binary

def show_img(img, scale=1, title='Image'):
    # cv.namedWindow('image', cv.WINDOW_NORMAL)
    h, w = img.shape[:2]
    img = cv.resize( img, (0,0), fx=scale, fy=scale  )
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()  



def show_imgs(imgs, scale=1, title='Image'):
    img = np.hstack(imgs)
    show_img(img, scale, title)

def get_loc(im, imt):
    result = cv.matchTemplate(im, imt, cv.TM_CCORR_NORMED)
    return cv.minMaxLoc(result)

def take_screenshot(rect):
    # 返回RGB图像
    hwnd = win32gui.FindWindow("UnityWndClass", "崩坏：星穹铁道")
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    rect[0] += left
    rect[1] += top 
    temp = pyautogui.screenshot(region=rect)
    screenshot = np.array(temp)
    screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2RGB)
    return screenshot

def take_minimap(rect = [47,58,187,187]):
    return take_screenshot(rect)

def take_fine_screenshot(rect = [47,58,187,187], n = 5, dt=0.01, dy=200):
    total = take_screenshot(rect)
    n = 5
    for i in range(n):
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0, -dy, 0, 0)
        mask = cv.compare(total, take_screenshot(rect), cv.CMP_EQ )
        total = cv.bitwise_and(total, mask )
        time.sleep(dt)
    time.sleep(0.1)
    for i in range(n):
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0, dy, 0, 0)
        mask = cv.compare(total, take_screenshot(rect), cv.CMP_EQ )
        total = cv.bitwise_and(total, mask )
        time.sleep(dt)
    minimap = cv.bitwise_and(total, mask )
    return minimap

def get_mask(img, color_range):
    lower, upper = color_range
    return cv.inRange(img, lower, upper)

def get_camera_fan(color = [130, 130, 60],angle=0, w=187, h=187, delta=90, dimen =3, radius= 90):
    center = (w//2, h//2)
    # radius = min(h, w)//2
    fan = np.zeros((h, w, dimen), np.uint8)
    # 计算圆心位置
    cx, cy = w // 2, h // 2
    axes = (w // 2, h // 2) 
    
    startAngle, endAngle = angle -45, angle +45 # 画90度

    cv.ellipse(fan, (cx, cy), axes, 0, startAngle, endAngle, color , -1)
    return fan

def get_gradient_mask(w,h):
    center = [w // 2, h // 2]
    radius = 0.8 *w
    # 创建渐变掩码
    gradient_mask = np.zeros((w, h), dtype=np.uint8)
    for r in range(gradient_mask.shape[0]):
        for c in range(gradient_mask.shape[1]):
            dist = np.sqrt((r-center[1])**2 + (c-center[0])**2)
            value =  max(0, min(1 - 2*dist/radius, 1))
            gradient_mask[r,c] = int(value * 255)
    return gradient_mask


def filter_contours_surround_point(gray, point):
    """过滤掉不包围指定点的轮廓"""
    contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # 过滤掉所有不包含指定点的轮廓
    filtered_contours = []
    for i in range(len(contours)):
        if cv.pointPolygonTest(contours[i], point, False) < 0:
            filtered_contours.append(contours[i])
            
    # 过滤掉所有不包围指定点的轮廓
    surrounded_contours = []
    for i in range(len(filtered_contours)):
        rect = cv.boundingRect(filtered_contours[i])
        if rect[0] <= point[0] <= rect[0] + rect[2] and \
           rect[1] <= point[1] <= rect[1] + rect[3]:
            surrounded_contours.append(filtered_contours[i])
            
    return surrounded_contours


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