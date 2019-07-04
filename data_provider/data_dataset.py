# -*- coding: utf-8 -*-
# @Time    : April 25
# @Author  : Alisdair Cameron, Xuyang SHEN
# @File    : data_dataset.py
# @IDE: PyCharm Community Edition

"""
    this is the tf.Dataset version for read data
"""
import json
import os
import tensorflow as tf

from data_provider.one_hot_coding import *


class TDataset:
    """
        instructions

    """

    def __init__(
            self,
            json_add,
            batch_size=4,
            num_epochs=10,
            buffer_size=150
    ):
        """

        :param json_add:
        :param batch_size:
        :param num_epochs:
        """
        self.absolute = json_add
        self.images_path_json = json_add + "images_path.json"
        self.labels_path_json = json_add + "labels_path.json"
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.buffer_size = buffer_size

        if not os.path.exists(self.images_path_json):
            raise ValueError("cannot find the json-storing image path: " + self.images_path_json)
        if not os.path.exists(self.labels_path_json):
            raise ValueError("cannot find the json-storing labels path: " + self.labels_path_json)
        self.initial()

    def initial(self):
        """
            import the image paths and ground_truth paths, to store in the memory
            in the meanwhile, check whether exits the image and ground_truth
        :return: None
        """
        name_check = []
        self.data_path = []
        self.gtruth_path = []

        # decode json to get address
        with open(self.images_path_json, 'r') as file:
            for line in file:

                # get info as dict format
                jsrd = json.loads(line)

                # verify the json
                if len(jsrd) != 2:
                    # not enough info of json
                    raise Exception("json format wrong -- image_path! " + jsrd)

                name_check.append(jsrd['name'])
                path = self.absolute + jsrd['path']
                self.data_path.append(path)

                if not os.path.exists(path):
                    raise ValueError("cannot find image: " + path)
        file.close()

        # decode json to get address
        with open(self.labels_path_json, 'r') as file:
            count = 0
            for line in file:

                # get info as dict format
                jsrd = json.loads(line)

                # verify the json
                if len(jsrd) != 2:
                    # not enough info of json
                    raise Exception("json format wrong I -- label_path! " + jsrd)

                if jsrd['name'] != name_check[count]:
                    raise Exception("json format wrong II -- label_path! " + jsrd)

                path = self.absolute + jsrd['path']
                self.gtruth_path.append(path)

                if not os.path.exists(path):
                    raise ValueError("cannot find label: " + path)

                count += 1

        file.close()

        print("All addresses(images and labels) have imported into memory")

    def dataset_input_fn(self):

        def _parse_function(filename):
            """
                decode images
            """
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image_resized = tf.image.resize_images(image_decoded, [512, 512])

            return image_resized

        def _parse_function2(example_proto):
            """
                decode tfRecords
            """
            features = tf.parse_single_example(example_proto,
                                               features={'data': tf.VarLenFeature(tf.int64)})
            data = features['data'].values
            # layer1 = tf.reshape(layer1, [512 * 512, 1])
            # a = tf.ones(
            #     shape=[512 * 512, 1],
            #     dtype=tf.int64
            # )
            # layer2 = tf.math.subtract(
            #     a, layer1
            # )
            # data = tf.concat([layer1, layer2], 1)
            data = tf.reshape(data, [512, 512])
            return data

        # transfer into tensor objects ([path1, path2, ...])
        data_names = tf.constant(np.array(self.data_path))
        gtruth_names = tf.constant(np.array(self.gtruth_path))

        # decode training set (images)
        trainset = tf.data.Dataset.from_tensor_slices(data_names)
        trainset = trainset.map(_parse_function)

        # decode labesl set (ground truth)
        label = tf.data.TFRecordDataset(
            filenames=gtruth_names
        )
        label = label.map(_parse_function2)

        dataset = tf.data.Dataset.zip((trainset, label))

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.repeat(self.num_epochs)
        print("DataSet size: ", dataset)

        iterator = dataset.make_one_shot_iterator()

        sample, labels = iterator.get_next()
        return sample, labels
