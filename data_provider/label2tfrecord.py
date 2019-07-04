# -*- coding: utf-8 -*-
# @Time    : May 11
# @Author  : Xuyang SHEN
# @File    : label2tfrecord.py
# @IDE: PyCharm CE Edition

import tensorflow as tf
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

from data_provider.one_hot_coding import *
from data_provider.line_connect import *


class TfGenerator:

    def __init__(self, path, js_name="label_data.json",
                 image_size=(512, 512),
                 overwrite=False):
        """
            please set up the file path until the train_set/
            e.g.
                path: 'E:/LaneDetection/train_set/'
        :param path: fold path where 'train_set'
        :param js_name: name of the json file
        """

        # pre-store
        self.store_images_path = []
        self.store_labels_path = []

        # import path
        self.absolute = path
        self.json_file = os.path.join(self.absolute, js_name)
        self.label_path = os.path.join(self.absolute, "labels/")

        # check path
        if not os.path.exists(self.absolute):
            raise Exception("Invalid path value")
        if not os.path.exists(self.json_file):
            raise Exception("Invalid json file")
        if not os.path.exists(self.label_path):
            raise Exception("Please create 'labels' folder under the train_set dir")

        # set property
        self.iSize = image_size
        self.overwrite = overwrite

    def generate_json(self):
        """
            it will generate the json files which includes all the files' paths
        :return: None
        """
        with open(self.absolute + "images_path.json", 'w') as file:
            cot = 0
            for item in self.store_images_path:
                json.dump(item, file)
                if cot < self.counter:
                    file.write("\n")
                cot += 1

        print("all images' name have been store into json. please check: ", self.absolute + "images_path.json")
        file.close()

        with open(self.absolute + "labels_path.json", 'w') as file:
            cot = 0
            for item in self.store_labels_path:
                json.dump(item, file)
                if cot < self.counter:
                    file.write("\n")
                cot += 1

        print("all labels' name have been store into json. please check: ", self.absolute + "labels_path.json")
        file.close()

    @staticmethod
    def _int64list_feature(value):
        """
            train feature generator
        :param value:
        :return:
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def write_tfReocrd(self, file_name, label):
        """
            write into tfRecord
        :param file_name:
        :param label:
        :return:
        """
        # create file path
        file_path = os.path.join(self.label_path + file_name + '.tfrecords')

        # it can continue last unfinished writing
        if os.path.exists(file_path):
            if not self.overwrite:
                print('Warning: ', file_name, '.tfrecords exist, and overwrite is set as False')
                return None

        writer = tf.python_io.TFRecordWriter(file_path)

        example = tf.train.Example(features=tf.train.Features(
            feature={'data': self._int64list_feature(label)}))

        writer.write(example.SerializeToString())
        writer.close()

    def run(self):
        self.starter = time.time()
        self.counter = 0
        print("program begins to write file, at", time.ctime())

        all_file_names = set()
        with open(self.json_file, 'r') as file:

            for line in file:

                # get info as dict format
                jsrd = json.loads(line)

                # verify the json
                if len(jsrd) < 3:
                    # not enough info of json
                    continue

                path = self.absolute + jsrd['raw_file']

                # get label
                rows = jsrd['lanes']
                col = jsrd['h_samples']
                coordinates = [[(x, y) for (x, y) in zip(row, col) if x >= 0] for row in rows]
                coordinates = [fill_line(coord) for coord in coordinates if coord]

                # flat labels
                labels = one_hot_encode(shape=(720, 1280),
                                        lst_coord=coordinates)
                labels = label_resize(
                    shape=(512, 512),
                    labels=labels,
                    recover=True,
                    recover_value=1
                )
                labels = labels.astype('int32')

                # must flat
                labels = np.reshape(labels, 512 * 512)

                # create unique file name
                file_name = path.split('/')
                file_name = file_name[-3:-1]
                file_name = "-".join(file_name)

                if file_name in all_file_names:
                    print("Warning: duplicate file names")
                else:
                    all_file_names.add(file_name)

                # convert
                self.write_tfReocrd(file_name, labels)

                # prepare to write in json
                store_image = dict()
                store_image['name'] = file_name
                store_image['path'] = jsrd['raw_file']

                store_label = dict()
                store_label['name'] = file_name
                store_label['path'] = 'labels/' + file_name + '.tfrecords'

                self.store_images_path.append(store_image)
                self.store_labels_path.append(store_label)

                self.counter += 1

                if self.counter % 50 == 0:
                    print(self.counter, " tfrecords have been created.", " It takes ",
                          "%0.2f" % (time.time() - self.starter), 'seconds')

        print("all labels have been writen into tfrecords. please check: ", self.label_path)
        file.close()

        # accurate the counter
        self.counter -= 1
        self.generate_json()
        print('all files has been converted. ', " It takes ", "%0.2f" % (time.time() - self.starter),
              'seconds to write ', self.counter, 'files')


if __name__ == '__main__':
    tusimple = TfGenerator(
        # path="/Volumes/RYAN/LaneDetection/train_set/",
        path="E:/LaneDetection/train_set/",
        js_name="total_label.json"
    )

    tusimple.run()
