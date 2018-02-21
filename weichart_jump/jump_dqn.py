import tensorflow as tf
import cv2
import sys
sys.path.append("image/")
import wrapped_flappy_bird as
import random
import numpy as np
from collections import deque


GAME = 'jump'












def wieght_variable(shape):
    initial = tf.truncated_normal(shape,0.01)
