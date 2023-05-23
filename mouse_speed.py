import ctypes
import json
import time
import win32api
import win32con
import win32gui
import win32print
from tools.log import log

class mouse_speed :
    # Define the distance and duration of the mouse movement

    def offset_divider (self) -> float:
        print("Start Testing , keep mouse on main screen")
        time.sleep(2)
        dx = 960
        time.sleep(0.5)
        win32api.SetCursorPos((0, 0))
        time.sleep(0.5)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, 0, 0, 0)
        x,y = win32api.GetCursorPos()
        print("鼠标位移期望值 ："  + str(dx))
        print("鼠标位移实际值 " + str(x))
        offset_d = (x/dx)
        print("The offset factor d is " + str(offset_d))
        return offset_d
    def adjust_mouse_speed (self) :
        offset = self.offset_divider()
        scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
        hwnd = win32gui.GetForegroundWindow()  # 根据当前活动窗口获取句柄
        log.info(hwnd)
        Text = win32gui.GetWindowText(hwnd)
        log.info(Text)

        # 获取活动窗口的大小
        window_rect = win32gui.GetWindowRect(hwnd)
        width = window_rect[2] - window_rect[0]
        height = window_rect[3] - window_rect[1]

        # 获取当前显示器的缩放比例
        dc = win32gui.GetWindowDC(hwnd)
        dpi_x = win32print.GetDeviceCaps(dc, win32con.LOGPIXELSX)
        dpi_y = win32print.GetDeviceCaps(dc, win32con.LOGPIXELSY)
        win32gui.ReleaseDC(hwnd, dc)
        scale_x = dpi_x / 96
        scale_y = dpi_y / 96

        diff = 930 * scale_x * 1.5625 / offset
        print("The diff is " + str(diff) )
        data = {"movement": diff }
        with open("temp\\Simulated_Universe\\mouse_speed.json", "w+") as outfile:
            json.dump(data, outfile)

if __name__ == '__main__':
    ms = mouse_speed()
    ms.adjust_mouse_speed()