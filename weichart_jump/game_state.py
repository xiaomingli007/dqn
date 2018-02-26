# -*- coding:utf-8 -*-

import cv2
import wda
import numpy as np
import time


SEC_PER_DU = 0.11

class gameState:
    def __init__(self):
        self.client = wda.Client('http://192.168.1.103:8100')
        self.sess = self.client.session()
    def find_head(self,img):
        # find small circles
        img_tmp = img.copy()
        img_tmp = cv2.resize(img_tmp,(540,960))
        head_found = False
        trial_cnt = 0
        while (head_found == False):
            # processing img
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(img_tmp, cv2.MORPH_CLOSE, kernel)
            edges = cv2.Canny(closed, 200, 200)  # use full colour img for canny
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 2, 180 - trial_cnt * 10,
                                       param1=100, param2=65, minRadius=12, maxRadius=18)
            gray = cv2.cvtColor(closed, cv2.COLOR_RGB2GRAY)

            # if nothing found, will return None, then try a few time.
            if (circles is not None):
                head_found = True
            else:
                if (trial_cnt > 5):  # return 0 as error.
                    return 0, 0, 0
            trial_cnt += 1

        circles = np.uint16(np.around(circles))

        # circle to return
        xo = 0
        yo = 0
        ro = 0

        for i in circles[0, :]:
            x = i[0]
            y = i[1]
            r = i[2]

            # find the head
            if (r < 25 and y > 380 and y < 520):

                if ((y < yo or yo is 0)):  # 寻找位置最靠上的那个源（有时候身子也会被识别成圆）
                    xo = x
                    yo = y
                    ro = r
        print '(%s,%s,%s)'%(xo,yo,ro)
        return xo, yo, ro

    def get_screenshot(self):
        self.client.screenshot('image/jump_tmp.png')


    def start(self,x=207,y=609):
        terminal = False
        reward = 1
        # print 'x=%s,y=%s'%(x,y)
        self.sess.tap(x=x, y=y)
        img = cv2.imread('image/jump_tmp.png')
        # print img.shape
        pos = self.find_head(img)
        if max(pos) == 0:
            terminal = True
            reward = -1

        x_t = cv2.cvtColor(cv2.resize(img, (84, 84)), cv2.COLOR_BGR2GRAY)
        s_t = x_t[:, :, np.newaxis]

        return s_t, reward, terminal

    def jump_step(self,input_actions):

        terminal = False
        reward = 1
        action_index = np.argmax(input_actions)

        tap_time = SEC_PER_DU*(action_index+1)

        img_start = cv2.imread('image/jump_tmp.png')
        start_pos = self.find_head(img_start)
        if max(start_pos) == 0:
            self.start()

        print 'action_index=%s,tap_time=%s'%(action_index,tap_time)
        self.sess.tap_hold(x=207,y=609,duration=tap_time)
        time.sleep(3)
        self.get_screenshot()
        img = cv2.imread('image/jump_tmp.png')
        # print img

        pos=self.find_head(img)
        if max(pos) == 0:
            terminal = True
            reward = -1

        x_t = cv2.cvtColor(cv2.resize(img, (84, 84)), cv2.COLOR_BGR2GRAY)
        s_t = x_t[:, :, np.newaxis]

        return s_t,reward,terminal






