# -*- coding: GBK -*-

#encoding:utf-8

#画9点-圆点
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img=np.zeros((1080,1920,3), np.uint8)
# cv2.circle(img,(3,3), 3, (255,255,255), -1)
# cv2.circle(img,(3,540), 3, (255,255,255), -1)
# cv2.circle(img,(3,1077), 3, (255,255,255), -1)
# cv2.circle(img,(960,3), 3, (255,255,255), -1)
# cv2.circle(img,(960,540), 3, (255,255,255), -1)
# cv2.circle(img,(960,1077), 3, (255,255,255), -1)
# cv2.circle(img,(1917,3), 3, (255,255,255), -1)
# cv2.circle(img,(1917,540), 3, (255,255,255), -1)
# cv2.circle(img,(1917,1077), 3, (255,255,255), -1)
#
#
#
# plt.imshow(img)
#
# plt.show()
#
# cv2.imwrite("C:\\Users\\my2b309\\Desktop\\pattern\\123.png",img)
# #
#
#
# #颜色亮度检测位置 9点
#
import cv2
import numpy as np
from matplotlib import pyplot as plt

img1=np.zeros((1080,1920,3), np.uint8)

# for i in range (1080):
#     for j in range (1920):
#         img1[i,j,0]=255
#         img1[i, j, 1] = 255
#         img1[i, j, 2] = 255
cv2.circle(img1,(320,180), 40, (255,255,255), -1)
cv2.circle(img1,(320,540), 40, (255,255,255), -1)
cv2.circle(img1,(320,900), 40, (255,255,255), -1)
cv2.circle(img1,(960,180), 40, (255,255,255), -1)
cv2.circle(img1,(960,540), 40, (255,255,255), -1)
cv2.circle(img1,(960,900), 40, (255,255,255), -1)
cv2.circle(img1,(1600,180), 40, (255,255,255), -1)
cv2.circle(img1,(1600,540), 40, (255,255,255), -1)
cv2.circle(img1,(1600,900), 40, (255,255,255), -1)

cv2.circle(img1,(96,54), 40, (255,255,255), -1)
cv2.circle(img1,(1824,54), 40, (255,255,255), -1)
cv2.circle(img1,(96,1026), 40, (255,255,255), -1)
cv2.circle(img1,(1824,1026), 40, (255,255,255), -1)

plt.imshow(img1)

plt.show()

cv2.imwrite("C:\\Users\\my2b309\\Desktop\\1234.png",img1)
####################
#######################
import cv2
import numpy as np
from matplotlib import pyplot as plt

# img1=np.zeros((1080,1920,3), np.uint8)
#
# for i in range (1080):
#     for j in range (1920):
#         img1[i,j,0]=0
#         img1[i, j, 1] = 0
#         img1[i, j, 2] = 255
#
#
# plt.imshow(img1)
#
# plt.show()
#
# cv2.imwrite("C:\\Users\\my2b309\\Desktop\\pattern\\B.png",img1)

################################################计算色坐标

# sR, sG and sB (Standard RGB) input range = 0 ÷ 255
# X, Y and Z output refer to a D65/2° standard illuminant
#
# sR=100
# sG=100
# sB=100
#
# var_R = ( sR / 255 )
# var_G = ( sG / 255 )
# var_B = ( sB / 255 )
#
# if ( var_R > 0.04045 ):
#     var_R = ( ( var_R + 0.055 ) / 1.055 )**2.4
# else:
#     var_R = var_R / 12.92
# if ( var_G > 0.04045 ):
#     var_G = ( ( var_G + 0.055 ) / 1.055 )**2.4
# else:
#     var_G = var_G / 12.92
# if ( var_B > 0.04045 ):
#     var_B = ( ( var_B + 0.055 ) / 1.055 )**2.4
# else:
#     var_B = var_B / 12.92
#
# var_R = var_R * 180
# var_G = var_G * 50
# var_B = var_B * 20
#
# X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
# Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
# Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
#
# print(X,Y,Z)
#
# Y = Y
# x = X / ( X + Y + Z )
# y = Y / ( X + Y + Z )
#
# print(x,y)


# 彩色图像转灰度图
#encoding:utf-8
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt



# cv2.putText(img, str(i), (123,456)), font, 2, (0,255,0), 3)
# 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细

# cv2.putText(grayimg, str(grayimg.mean()), (100, 100), cv2.FONT_HERSHEY_SIMPLEX , 8, (128, 0, 128), 8)

def gray_mean(img):

    height = img.shape[0]
    width = img.shape[1]
    grayimg = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            gray = 0.30 * img[i, j][0] + 0.59 * img[i, j][1] + 0.11 * img[i, j][2]
            grayimg[i, j] = np.uint8(gray)
    print(round(img.mean(), 2))
    return round(img.mean(),2)
def puttext(dstimg,dststr,a,b):
    cv2.putText(dstimg, dststr, (a, b), cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 0, 128), 4)



#计算中心点灰阶



#读取中心图像

def graycenter(imgno):
    img = cv2.imread("C:\\Users\\my2b309\Pictures\\GainExpo\\"+str(imgno)+".bmp")
    height = img.shape[0]
    width = img.shape[1]
    grayimg = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            gray = 0.30 * img[i, j][0] + 0.59 * img[i, j][1] + 0.11 * img[i, j][2]
            grayimg[i, j] = np.uint8(gray)

    print(imgno,round(img.mean(), 2))
    return round(img.mean(), 2)

# for i in range (2,16):
#     graycenter(1000*i)
#
# graycenter(16667)

def colorcenter(imgname):
    img = cv2.imread("C:\\Users\\my2b309\Pictures\\"+str(imgname)+".bmp")
    img = img[519:569, 703:753]
    height = img.shape[0]
    width = img.shape[1]
    sum_r = 0
    sum_g = 0
    sum_b = 0
    count = 0
    for i in range(height):
        for j in range(width):
            sum_b = sum_b + img[:, :, 0].mean()
            sum_g = sum_g + img[:, :, 1].mean()
            sum_r = sum_r + img[:, :, 2].mean()
            count+=1
    grayR= sum_r/count
    grayG= sum_g/count
    grayB= sum_b/count
    var_R = ( grayR / 255 )
    var_G = ( grayG / 255 )
    var_B = ( grayB / 255 )

    if ( var_R > 0.04045 ):
        var_R = ( ( var_R + 0.055 ) / 1.055 )**2.4
    else:
        var_R = var_R / 12.92
    if ( var_G > 0.04045 ):
        var_G = ( ( var_G + 0.055 ) / 1.055 )**2.4
    else:
        var_G = var_G / 12.92
    if ( var_B > 0.04045 ):
        var_B = ( ( var_B + 0.055 ) / 1.055 )**2.4
    else:
        var_B = var_B / 12.92

    var_R = var_R * 100
    var_G = var_G * 100
    var_B = var_B * 100

    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505


    Y = Y
    x = X / ( X + Y + Z )
    y = Y / ( X + Y + Z )

    print(imgname,X,Y,Z,x,y)


# colorcenter(10)
# colorcenter(15)
# colorcenter(35)
# colorcenter(40)
#
# for i in range(10,41):
#     colorcenter(5*i)
# colorcenter(210)
# colorcenter(220)
# colorcenter(230)
# colorcenter(240)
# colorcenter(250)


# for i in range(1,13):
#     colorcenter(i)


# img1=img[156:206,219:269]
# img2=img[156:206,703:753]
# img3=img[156:206,1187:1237]
# img4=img[519:569,219:269]
# img5=img[519:569,703:753]
# img6=img[519:569,1187:1237]
# img7=img[882:932,219:269]
# img8=img[882:932,703:753]
# img9=img[882:932,1187:1237]



# puttext(img,str(gray_mean(img1)),244,181)
# puttext(img,str(gray_mean(img2)),728,181)
# puttext(img,str(gray_mean(img3)),1212,181)
# puttext(img,str(gray_mean(img4)),244,544)
# puttext(img,str(gray_mean(img5)),728,544)
# puttext(img,str(gray_mean(img6)),1212,544)
# puttext(img,str(gray_mean(img7)),244,907)
# puttext(img,str(gray_mean(img8)),728,907)
# puttext(img,str(gray_mean(img9)),1212,907)

# cv2.imwrite("C:\\Users\\my2b309\\Desktop\\color\\131-1.bmp",img)
# #显示图像
# cv2.imshow("src", img)
# # cv2.imshow("gray", grayimg)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#
# img=cv2.imread("C:\\Users\\my2b309\\Desktop\\123\\Air VID.png")
# cv2.circle(img,(959,539), 60, (255,255,255), -1)
#
# img[365:475,905:1015]=img[5:115,230:340]
# img[605:715,905:1015]=img[5:115,230:340]
# img[485:595,680:790]=img[5:115,230:340]
# img[485:595,1130:1240]=img[5:115,230:340]
#
# img[510:570,930:990]=img[30:90,255:315]
#
# cv2.imshow("src", img)
# # # cv2.imshow("gray", grayimg)
# # #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# cv2.imwrite("C:\\Users\\my2b309\\Desktop\\a.png",img)