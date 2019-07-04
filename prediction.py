# -*- coding: utf-8 -*-
# @Time    : May 25
# @Author  : Xuyang SHEN, Alisdair Cameron, Xinqi Zhu
# @File    : prediction.py
# @IDE: PyCharm Community Edition

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imageio

from enet.ENet import *
from config import *
from data_provider.one_hot_coding import one_hot_decode


def transparent_circle(img, center, radius, rgb_alpha):
    center = tuple(map(int, center))
    rgb = rgb_alpha[:3]
    alpha = rgb_alpha[-1]
    radius = int(radius)

    roi = slice(center[1] - 3, center[1] + 3), slice(center[0] - 3, center[0] + 3)
    overlay = img[roi].copy()
    cv2.circle(img, center, radius, rgb, thickness=0, lineType=cv2.LINE_AA)
    cv2.addWeighted(src1=img[roi], alpha=alpha, src2=overlay, beta=1. - alpha, gamma=0, dst=img[roi])


def display_plot(origin, one_hot, comb):
    fig = plt.figure(figsize=(10, 33))
    ax1 = fig.add_subplot(311)
    origin = np.reshape(origin, (512, 512, 3))
    origin = origin.astype('uint8')
    ax1.imshow(origin)
    ax1.set_title("origin image")

    ax2 = fig.add_subplot(312)
    ax2.imshow(one_hot, cmap='Greys')
    ax2.set_title("one_hot image")

    ax2 = fig.add_subplot(313)
    ax2.imshow(comb)
    ax2.set_title("combine the one_hot into original image")

    plt.show()


# ---------------------------------------------------------
# run parser
# ---------------------------------------------------------
config = parse_cmd_testing_args()

test_add = config.test_dataset
model_add = config.model
output_add = config.result_address
display = config.display

# ---------------------------------------------------------
# initialized the model
# ---------------------------------------------------------
model_address = os.path.join(os.getcwd(), model_add)
lane_detect = tf.estimator.Estimator(
    model_fn=ENet,
    model_dir=model_address,
)

# ---------------------------------------------------------
# prediction data set
# ---------------------------------------------------------
pred_images = []
filename = os.listdir(test_add)

for f in filename:
    im = plt.imread(os.path.join(test_add, f))
    eval_data = cv2.resize(im, (512, 512))

    eval_data = eval_data.astype('float32')
    eval_data = np.reshape(eval_data, (1, 512, 512, 3))
    pred_images.append(eval_data)
pred_images = np.array(pred_images)

# ---------------------------------------------------------
# prediction the model
# ---------------------------------------------------------
pred_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=pred_images,
    shuffle=False,
    batch_size=1
)
pred_result = lane_detect.predict(
    input_fn=pred_input_fn,
    checkpoint_path=None
)

# iteration all prediction
var = 0
for i in pred_result:
    i = i['classes']

    re = np.zeros(shape=(512, 512))
    for col in range(512):
        for row in range(512):

            # focus on the probability
            if i[col, row, 0] < i[col, row, 1]:
                re[col, row] = 255

    # one_hot labels
    re = np.reshape(re, (512, 512))
    re = re.astype('uint8')

    path = output_add + "pred-" + filename[var]
    imageio.imwrite(path, re)

    # combine original with current
    coords = one_hot_decode(re)

    im = pred_images[var]
    im = np.reshape(im, (512, 512, 3))
    im = im.astype('uint8')
    for coord in coords:
        transparent_circle(img=im, center=coord, radius=1, rgb_alpha=[102, 255, 51, 0.2])

    path = output_add + "comb-" + filename[var]
    imageio.imwrite(path, im)

    if display == 'yes':
        display_plot(
            pred_images[var], re, im
        )
    var += 1
