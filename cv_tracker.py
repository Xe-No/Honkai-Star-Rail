#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Created by Xe-No at 2023/5/17
# 利用cv识别小地图并返回地图索引与坐标
from cv_tools import *
import cv2 as cv
import time
import sys, os, glob
from tools.switch_window import switch_window as sw
from tools.get_angle import *
from tools.route_helper import *
from ray_casting import ray_casting
# import log
import win32api
import win32con
import pyautogui
import math
from PIL import Image


def match_scaled(img, template, scale):
    resized_template = cv.resize(template, (0,0), fx=scale, fy=scale)
    res = cv.matchTemplate(img, resized_template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return [max_val, max_loc]

def find_best_match(img, template, scale_range=(1.4, 2.0, 0.05)):
    best_match = None
    max_corr = -1
    print(img.shape)
    for scale in np.arange(scale_range[0], scale_range[1],scale_range[2]):
        [max_val, max_loc] = match_scaled(img, template, scale)
        print(f'正在匹配 {scale}，相似度{max_val}')
        if max_val > max_corr:
            max_corr = max_val
            max_ret = [scale, max_val, max_loc]
    return max_ret


class Tracker():
    """docstring for Tracker"""
    def __init__(self):
        self.map_prefix = './temp/Simulated_Universe/map_raw/'
        self.masked_map_prefix = './temp/Simulated_Universe/map_masked/'
        self.template_prefix = './temp/Simulated_Universe/map_template_sample/'
        self.result_prefix = './temp/Simulated_Universe/map_result/'

        # self.minimap_rect = [47,58,187,187] # 全尺度，只有圆形部分匹配度较差
        self.minimap_rect = [77,88,127,127]
        # 节省时间，只有需要时才调用load_all_masked_maps
        self.masked_maps = None


    def load_all_images(self, prefix, flag=cv.IMREAD_UNCHANGED):
        images = {}
        for file in glob.glob(f'{prefix}*.png'):
            index = os.path.basename(file)
            images[index] = cv.imread(file, flag)
        return images

    def load_all_gray_images(self, prefix):
        images = {}
        for file in glob.glob(f'{prefix}*.png'):
            index = os.path.basename(file)
            images[index] = cv.imread(file, cv.IMREAD_GRAYSCALE)
        return images

    def load_map(self, index, prefix):
        return cv.imread(prefix+index)

    def load_all_masked_maps(self):
        images = {}
        for file in glob.glob(f'{self.masked_map_prefix}*.png'):
            index = os.path.basename(file)
            images[index] = cv.cvtColor( cv.imread(file), cv.COLOR_BGR2GRAY)
        self.masked_maps = images
        return images

    def save_all_masked_maps(self):
        maps = self.load_all_images(self.map_prefix)
        # # 路面掩膜

        # show_img(b, 0.25)
        
        # map_b = get_mask(map_hsv,np.array([[0,0,30],[360,10,90]]))
        # map_b = cv.medianBlur(map_b, 5)
        # 路沿掩膜
        masked_maps = {}
        for index, map_r in maps.items():
            b, g, r, a = cv.split(map_r)
            mask = a>200
            b = b * mask
            map_b = cv.threshold(b, 20, 150, cv.THRESH_BINARY)[1]
            # map_hsv = cv.cvtColor(map_r, cv.COLOR_BGR2HSV)
            # map_s = get_mask(map_hsv,np.array([[0,0,180],[360,10,255]]))
            # map_s = cv.medianBlur(map_s, 3)
            masked_maps[index] = map_b
            cv.imwrite(self.masked_map_prefix + index, map_b)
        # 保存之后也返回
        return masked_maps

    def get_img(self, prefix=None, img_path=False ):
        if img_path:
            img_r = cv.imread(prefix + img_path)
        else:
            img_r = take_screenshot(self.minimap_rect)
        return img_r

    def get_minimap_mask(self, mini_r, color_range = np.array([[0,0,180],[360,10,255]]) ):
        # img_hsv = cv.cvtColor(img_r, cv.COLOR_BGR2HSV)
        # img_s = get_mask(img_hsv, color_range)
        mini_r = cv.cvtColor(mini_r, cv.COLOR_BGR2GRAY)
        mini_b = cv.threshold(mini_r, 20, 150, cv.THRESH_BINARY)[1]
        kernel = np.ones((5, 5), np.uint8)
        mini_b = cv.dilate(mini_b, kernel, iterations=1)

        return mini_b

    def get_coord_by_map(self, map_b, img_r, scale=2.09):
        # 固定地图，识别坐标 map_index 图片索引 img 彩色图
        img_s =self.get_minimap_mask(img_r)
        img_s = cv.resize(img_s, (0,0), fx = scale, fy = scale) # 小地图到大地图的缩放尺度
        w,h = img_s.shape[:2]
        min_val, max_val, min_loc, max_loc = get_loc(map_b, img_s)

        print(max_val)
        cx = max_loc[0] + w//2
        cy = max_loc[1] + h//2
        pos = [cx,cy]
        return [cx, cy, max_val]

    def get_front_angle(self):
        main_angle = get_angle()
        mini_r = take_fine_screenshot(self.minimap_rect, n=1, dy=30)
        mini_r = cv.cvtColor(mini_r, cv.COLOR_BGR2GRAY)
        mini_b = cv.threshold(mini_r, 20, 150, cv.THRESH_BINARY)[1]
        # 扇形组成简易神经网络
        h,w = mini_r.shape[:2]
        fans = {}
        fans['f'] = get_camera_fan(color = 255, angle=main_angle, w=w, h=h, delta=30, dimen=1, radius=60)
        fans['l'] = get_camera_fan(color = 255, angle=main_angle-60, w=w, h=h, delta=90, dimen=1, radius=60)
        fans['r'] = get_camera_fan(color = 255, angle=main_angle+60, w=w, h=h, delta=90, dimen=1, radius=60)
        # fans['b'] = get_camera_fan(color = 255, angle=main_angle-180, w=w, h=h, delta=90, dimen=1, radius=60)

        
        lx = np.linspace(-1, 1, w)
        ly = np.linspace(-1, 1, h) 
        xx, yy= np.meshgrid(lx,ly)
        rr = xx*xx++yy*yy
        count = {}
        for key, fan in fans.items():
            # cx = np.mean(xx * fan)
            # cy = np.mean(yy * fan)  
            count[key] = np.sum(mini_b * fan * rr)/255

        print(count)

        if count['f'] > 200:
            angle = 0
        else:
            if count['r'] > count['l']:
                angle =90
            else: 
                angle =-90

        # if count['l'] >100 or count['r'] >100:
        #     angle =   (count['r'] - count['l']) / (count['r'] + count['l']) *90
        #     angle =   min(90,max(-90,angle))
        # else: 
        #     # if count['br'] > count['bl']:
        #     #     angle = 90
        #     # else:
        #     #     angle = -90
        #     angle = 180




        # angle = np.degrees(np.arctan2(cy, cx))
        # angle = np.around(angle, 2) 
        return angle

    def find_map(self, img_r, scale=1.66):
        if self.masked_maps == None:
            self.load_all_masked_maps()
        max_index = -1
        max_corr = -1
        max_ret = None
        for index, map_b in self.masked_maps.items():
            # img_hsv = cv.cvtColor(img_r, cv.COLOR_BGR2HSV)
            # img_s = get_mask(img_hsv,np.array([[0,0,160],[360,10,255]]))

            img_s = self.get_minimap_mask(img_r)
            # show_img(img_s)
            print(f'正在匹配{index}的缩放尺度')
            # [scale, corr, loc] = find_best_match(map_b, img_s, (100,200,5))
            # [cx, cy, corr] = self.get_coord_by_map(self.masked_maps[index],img_r)
            [corr, loc] = match_scaled(map_b, img_s, scale)
            print(f'正在找{index}，相似度{corr}')
            # if corr < 0.5:
            #     continue
            if corr > max_corr:
                max_corr = corr
                max_ret = [index, corr, loc, scale]

            [index, corr, loc, scale] = max_ret
            [hh,hw] = img_r.shape[:2] # 半高半宽
            hh = int(hh * scale //2)
            hw = int(hw * scale //2)
            x = loc[0] + hw
            y = loc[1] + hh
            print(f'地图{index}的相似度为{corr},当前坐标为{[x,y]}')
        return [index, [x,y], [hw,hh] ,corr]

    def get_scale(self, map_b, mini_b ):

        # 获取小地图比例尺，一次获取，终生受益

        [scale, max_val, max_loc] = find_best_match(map_b, mini_b)
        # show_img(ret)
        [h, w] = mini_b.shape[:2]
        hf = int(h * scale)
        wf = int(w * scale)

        cv.rectangle(map_b, max_loc, np.add(max_loc, [wf,hf]), 255, 5   )
        show_img(map_b, 0.25)
        
        return scale



def test_1():

    tracker = Tracker()


    map_b = cv.imread(tracker.masked_map_prefix + '56.png', cv.IMREAD_GRAYSCALE)

    img_r = cv.imread(tracker.template_prefix+'screenshot_1684407906_26.png')
    img_b = tracker.get_minimap_mask(img_r, color_range = np.array([[0,0,180],[255,60,255]]))

    # img_g = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
    # map_g = cv.cvtColor(map_r, cv.COLOR_BGR2GRAY)
    # # img_g = map_g[400:900,400:900]
    # map_g = cv.resize(map_g, (0,0) ,fx=1.2,fy=1.2)
    img_b = map_b[1000:1600,1000:1600]
    ret = sift_match(map_b,img_b)
    show_img(ret, 0.5)


    time.sleep(0.5)
    sw()
    time.sleep(0.5)




def test_2(index):
    # 找某张图缩放比

    time.sleep(0.5)
    sw()
    time.sleep(0.5)
    tr = Tracker()
    
    map_bgra = cv.imread(tr.map_prefix + index, cv.IMREAD_UNCHANGED) 
    b, g, r, a = cv.split(map_bgra)
    mask = a>200
    b = b * mask
    # show_img(b, 0.25)
    map_b = cv.threshold(b, 20, 150, cv.THRESH_BINARY)[1]

    mini = take_fine_screenshot([77,88,127,127])
    show_img(mini)
    mini = cv.cvtColor(mini, cv.COLOR_BGR2GRAY)
    mini_b = cv.threshold(mini, 20, 150, cv.THRESH_BINARY)[1]
    print('?')
    kernel = np.ones((5, 5), np.uint8)
    mini_b = cv.dilate(mini_b, kernel, iterations=1)


    scale = tr.get_scale( map_b, mini_b)

    print(f'地图{index}最佳匹配缩放百分比为{scale}')

def test_3():
    # 将所有地图转为掩膜图
    tr = Tracker() 
    tr.save_all_masked_maps()

def test_4():
    # 找地图测试
    time.sleep(0.5)
    sw()
    time.sleep(0.5)
    tr = Tracker()



    img_r = take_fine_screenshot(tr.minimap_rect)
    [index, [x,y], [hw,hh] ,corr] = tr.find_map(img_r )
    map_r = tr.load_map(index, tr.map_prefix)




    print([x,y])
    print([hw,hh])
    cv.rectangle(map_r, [x,y], [x+1,y+1], [0,0,255],5 )
    cv.rectangle(map_r, [x-hw,y-hh], [x+hw,y+hh], [255,0,0],5 )
    show_img(img_r)
    show_img(map_r,0.25)

def test_5():
    time.sleep(0.5)
    sw()
    time.sleep(0.5)

    tr = Tracker()

    # pyautogui.keyDown('w')
    # for _ in range(20):
    #     front_angle = tr.get_front_angle()
    #     # char_angle = get_angle()
    #     # bias_angle = camera_angle-char_angle
    #     # bias_angle -= round((bias_angle)/360)*360
    #     # print(bias_angle)
    #     # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(bias_angle*20), 0, 0, 0)
    #     turn_by(front_angle, speed_factor = 5)
    #     time.sleep(0.3)
    # # print(get_camera_angle())

    # pyautogui.keyUp('w')
    # show_img(mask)

def test_6():
    time.sleep(0.5)
    sw()
    time.sleep(0.5)

    img = take_fine_screenshot()
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite( 'test.png', img)

def test_7():
    # 光线透射法暴力求最可能路径
    prefix = './temp/Simulated_Universe/'
    time.sleep(0.5)
    sw()
    time.sleep(0.5)
    tr = Tracker()

    while not get_image_pos(f'{prefix}atlas_portal.jpg', 0.9):
        image = take_fine_screenshot(tr.minimap_rect)
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        image =  cv2.dilate(image, kernel, iterations=1)
        image_b = get_binary(image, 30)
        angle_r = ray_casting(image_b, threshold=128, step=5)
        char_angle = get_angle()
        max_weight = -1
        max_angle_r = None
        final_angle = 0
        for angle, r in angle_r:
            # 根据角度对转向进行投票
            turn_angle = angle - char_angle
            turn_angle -= round(turn_angle/360)*360

            weight = np.sign(turn_angle) * r /60
            if abs(turn_angle) > 90:
                weight*=0.3
            final_angle  += weight * angle /18  


        # max_angle, max_r = max_angle_r
        print(f'角度{final_angle},半径{r}')
        r-=5
        if r>0 :
            turn_by(final_angle, speed_factor = 5.0)
            move(r/25)
            time.sleep(0.3)

        


if __name__ == '__main__':
    test_4()
    # test_6()
    index = '62.png'
    # test_2(index)
    # test_4()

