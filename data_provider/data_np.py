# -*- coding: utf-8 -*-
# @Time    : April 25
# @Author  : Xuyang SHEN
# @File    : data_np.py
# @IDE: PyCharm Community Edition

"""
    this is the numpy version for read data
"""

import matplotlib.pyplot as plt
import json
import os
from math import floor

from data_provider.one_hot_coding import *
from data_provider.line_connect import *


class TData:

    def __init__(self, path, js_name="label_data_0531.json",
                 image_size=(512, 512),
                 train_ratio=0.8,
                 label_flat=False):
        """
            please set up the file path until the train_set/
            e.g.
                path: 'E:/LaneDetection/train_set/'
        :param path: fold path where 'train_set'
        :param js_name: name of the json file
        """

        # pre-store
        self.eval_label = []
        self.eval_set = []
        self.train_label = []
        self.train_set = []

        # import path
        self.absolute = path
        self.json_file = os.path.join(self.absolute + js_name)

        # check path
        if not os.path.exists(self.absolute):
            raise Exception("Invalid path value")
        if not os.path.exists(self.json_file):
            raise Exception("Invalid json file")

        # set property
        self.iSize = image_size
        self.tRatio = train_ratio
        self.lFlat = label_flat

    def run(self):

        i = []
        l = []

        with open(self.json_file, 'r') as file:

            count = 0

            for line in file:
                # get info as dict format
                jsrd = json.loads(line)

                # verify the json
                if len(jsrd) < 3:
                    # not enough info of json
                    continue

                # read images
                path = self.absolute + jsrd['raw_file']
                im = plt.imread(path)
                im = cv2.resize(im, self.iSize)
                im = im.astype('float32')
                # store images
                i.append(im)

                # get label
                rows = jsrd['lanes']
                col = jsrd['h_samples']
                coordinates = [[(x, y) for (x, y) in zip(row, col) if x >= 0] for row in rows]
                coordinates = [fill_line(coord) for coord in coordinates if coord]

                # flat labels
                labels = one_hot_encode(shape=(720, 1280),
                                        lst_coord=coordinates)
                labels = label_resize(
                    shape=self.iSize,
                    labels=labels,
                    recover=True,
                    recover_value=1
                )
                labels = labels.astype('uint32')

                if self.lFlat:
                    labels = np.reshape(labels, self.iSize[0] * self.iSize[1])

                l.append(labels)

                count = count + 1

        index = floor(count * self.tRatio)
        self.train_set = np.asarray(i[0:index])
        self.train_label = np.asarray(l[0:index])
        self.eval_set = np.asarray(i[index:-1])
        self.eval_label = np.asarray(l[index:-1])

    def fetch_train_set(self):
        return self.train_set

    def fetch_train_labels(self):
        return self.train_label

    def fetch_eval_set(self):
        return self.eval_set

    def fetch_eval_labels(self):
        return self.eval_label


if __name__ == '__main__':

    tusimple = TData(
        # path="/Volumes/RYAN/train_set/",
        path="E:/LaneDetection/train_set/",
        js_name="label_data_0531.json"
    )

    tusimple.run()
    ts = tusimple.fetch_train_set()
    tl = tusimple.fetch_train_labels()


    for im, ls in zip(ts, tl):
        ls = one_hot_decode(ls)

        im = im.astype('uint8')

        for coord in ls:
            cv2.circle(im, center=coord, radius=2, color=(0, 255, 0))

        plt.imshow(im)
        plt.show()

