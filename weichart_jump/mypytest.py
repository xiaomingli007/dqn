# -*- coding:utf-8 -*-

import sys
import numpy as np
import cv2
import math
from PIL import ImageGrab


def find_head(img, output_img):
    # find small circles
    head_found = False
    trial_cnt = 0
    while(head_found == False):
        #processing img
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(closed, 200, 200)             # use full colour img for canny
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 2, 180 - trial_cnt *10,
                                   param1=100, param2=65, minRadius=12, maxRadius=18)
        gray = cv2.cvtColor(closed, cv2.COLOR_RGB2GRAY)

        # if nothing found, will return None, then try a few time.
        if(circles is not None):
            head_found = True
        else:
            if (trial_cnt > 5):     # return 0 as error.
                return 0, 0, 0
        trial_cnt += 1

    circles = np.uint16(np.around(circles))

    # circle to return
    xo=0
    yo=0
    ro=0

    # find and draw  # 遍历所有找到的
    for i in circles[0, :]:
        x = i[0]
        y = i[1]
        r = i[2]

        # find the head
        if (r < 25 and y > 380 and y < 520 ):
            # 画出来
            cv2.circle(output_img, (i[0], i[1]), i[2], (128, 0, 255), 2)  # outer
            cv2.circle(output_img, (i[0], i[1]), 2, (128, 0, 255), 3)  # centre

            if((y < yo or yo is  0 )): # 寻找位置最靠上的那个源（有时候身子也会被识别成圆）
                xo=x
                yo=y
                ro=r
        # not the head #
        else:
            # 不是头，用另一种颜色画出来
            cv2.circle(output_img, (i[0], i[1]), i[2], (127, 127, 0), 2)
            cv2.circle(output_img, (i[0], i[1]), 2, (127, 127, 0), 3)

    # 判断头位置的辅助线
    cv2.line(output_img, (0, 380), (540, 380), (128, 0, 255), 1)
    cv2.line(output_img, (0, 530), (540, 530), (128, 0, 255), 1)

    cv2.imshow("Head", edges)
    return xo, yo, ro

def draw_loc(imgin, loc, color = (0, 0, 255)):
    loc = np.uint16(np.around(loc))
    cv2.putText(imgin, '(' + str(loc[0]) + ',' + str(loc[1]) + ")", (loc[0] + 30, loc[1] +30), cv2.FONT_HERSHEY_PLAIN ,
                2, color, 2)
    return


def draw_text(imgin, loc, text):
    loc = np.uint16(np.around(loc))
    cv2.putText(imgin, text, (loc[0] + 10, loc[1] +30), cv2.FONT_HERSHEY_PLAIN ,
                2, (255, 125, 127), 2)
    return

def find_foot_loc(x,y,r):
    xo = x
    yo = y + 75 # distance between head to foot
    return (xo, yo)


def find_target_loc(img, output_img, loc_foot, index = 0):
    edges = cv2.Canny(img, 15, 70)
    points = cv2.findNonZero(edges)   # find all none zero point
    #print(points)
    for point in points:
        x = point[0][0]
        y = point[0][1]

        if y < 250:
            continue
        if np.abs(x - loc_foot[0]) < 30: # horizontal distance to foot
            continue
        if loc_foot[1] - y < 30:   # higher than foot
            continue
        if x < 50 or x > 540 - 50: # too close to bondary
            continue
        break

    # Find top tips of the target
    xo = x              # 0~33步 是大图形，偏移量比较多
    yo = y + 45

    if index > 33:      # 33步~75步 是中等图形， 偏移量减小
        yo = y + 35
    if index > 75:      # 75 步以上 都是非常小的图形比较多， 偏移量最小
        yo = y + 20

    cv2.imshow('Canny Line', edges)
    return (xo, yo)


def main():
    img = cv2.imread('./tiao.png')
    img = cv2.resize(img, (540, 960))  # 统一转换成 540*960 分辨率

    # print(img.shape)
    step = fault_cnt = 1

    output_img = img.copy()  # 复制一张图用来输出

    i= find_head(img,output_img)
    # loc_foot = find_foot_loc(i[0], i[1], i[2])  # 找到小人脚部的位置
    #
    # # find (and draw) target
    # loc_target = find_target_loc(img, output_img, loc_foot, step)  # 找落点
    #
    # # calculate distance
    # dist = np.sqrt((loc_target[0] - loc_foot[0]) * (loc_target[0] - loc_foot[0])  # 三角函数咯，计算距离
    #                + (loc_target[1] - loc_foot[1]) * (loc_target[1] - loc_foot[1]))
    #
    # # calculate pressing time
    # time = dist / 365  # 魔法参数2
    # time = dist / 385 + 0.03
    # time = dist / 375 + 0.01

    # -------- Draw informations on output image ---------------
    # me and target
    # draw_loc(output_img, loc_foot, (255, 0, 0))  #
    # cv2.circle(output_img, loc_target, 3, (255, 0, 255), 3)  # target
    # draw_loc(output_img, loc_target, (255, 0, 0))
    #
    # # draw line between me and target
    # cv2.line(output_img, loc_foot, loc_target, (255, 0, 0), 3)  # 小人位置跟目标落点之间画一条线
    #
    # # print information on output image
    # draw_text(output_img, (50, 200), 'Distance  :' + str(np.int(dist)) + 'pix')  # 距离
    # draw_text(output_img, (50, 250), 'Press time:' + str(np.int(time * 1000)) + 'ms')  # 时间
    # draw_text(output_img, (50, 300), 'Faults     :' + str(fault_cnt))  # 错误次数
    #
    # # print step
    # cv2.putText(output_img, 'STEP:' + str(step), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  # 第几步了
    #             1, (255, 0, 225), 2)

    # cv2.imshow("gray", gray)
    print i
    cv2.imshow('OUTPUT', output_img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()













