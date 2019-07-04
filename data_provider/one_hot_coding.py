# -*- coding: utf-8 -*-
# @Time    : April 25
# @Author  : Xuyang SHEN
# @File    : one_hot_coding.py
# @IDE: PyCharm Community Edition

"""
    this is the one hot encoding format
"""

import numpy as np
import cv2


def one_hot_encode(lst_coord,
                   shape: tuple = (720, 1280)):
    """

    :param shape:
    :param lst_coord:
    :return:
    """
    re = np.zeros(shape)
    for coord in lst_coord:
        for c in coord:
            try:
                re[c[1], c[0]] = 1
            except IndexError:
                print('Index out of bound for coordinates: ', c)
    return re


def one_hot_decode(coding):
    """

    :param coding:
    :return:
    """
    re = []
    for col in range(coding.shape[0]):
        for row in range(coding.shape[1]):
            value = coding[col, row]
            if value > 0:
                re.append((row, col))
            elif value == 0:
                pass
            else:
                print("Error value: ", value, ' with coord', (row, col))
    return re


def labels_recover(labels, value=1):
    """

    :param labels:
    :param value:
    :return:
    """
    labels[labels > 0] = value
    return labels


def label_resize(shape: tuple, labels,
                 recover=False, recover_value=1):
    """

    :param shape:
    :param labels:
    :param recover:
    :param recover_value:
    :return:
    """
    labels = cv2.resize(labels, shape)
    if recover:
        return labels_recover(labels, recover_value)
    else:
        return labels



def one_element_right(labels):
    """

    :param labels:
    :return:
    """
    re = []
    for x in labels:
        for i in range(0, len(labels[0])):
            if x[i] == 1:
                re.append(i)
    return re
