from cv_tools import *

# 灰色

img_r = cv.imread('img_r.jpg')
img_hsv = cv.cvtColor(img_r, cv.COLOR_BGR2HSV)

img_b = get_binary(img_r,180)

mask_gray = get_mask(img_hsv,np.array([[0,0,255*0.15],[0,0,255*0.25]]))
mask_cyan = get_mask(img_hsv,np.array([[70,0,0],[110,255,255]]))




mask_white = get_mask(img_hsv,np.array([[0,0,170],[360,30,255]]))
edge = cv.Canny(img_hsv, 600, 800)
greater_mask = (img_r[:,:,1]<img_r[:,:,2])
h_mask = (img_r>70) *(img_r<110) * 1.0
print(h_mask)

print(greater_mask)

print(img_hsv[92,94])
print(img_hsv[66,71])
# show_imgs([mask_gray,mask_cyan,mask_white,edge,img_b])
show_img(greater_mask)