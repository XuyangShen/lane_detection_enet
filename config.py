# -*- coding: utf-8 -*-
# @Time    : May 25
# @Author  : Xuyang SHEN
# @File    : config.py
# @IDE: PyCharm Community Edition

# import tensorflow as tf

import argparse


def parse_cmd_training_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", dest ="trainset_address", metavar="INPUT-PATH",
                        default="sample_data/train_set/",
                        help="The path to the train_set (include the train_set). (default: %(default)s)")
    parser.add_argument("-m", dest="model_add", metavar="MODEL-PATH", default="tmp/train_01/",
                        help="The path to store the training model. (default: %(default)s)")
    parser.add_argument("-l", dest="offline_label_generator", metavar="yes/no",
                        choices=["yes", "no"], default="no",
                        help="if it is the first time to run the train, please select yes")
    parser.add_argument("-gt", "--ground_truth", dest="ground_truth", metavar="PATH", default=None,
                        help="input the json file of the labels")
    parser.add_argument("-e", "--epochs", dest="num_epochs", metavar="iteration", type=int, default=40,
                        help="input the number of epochs need to run (default: %(default)s)")
    parser.add_argument("-b", "--batch_size", dest="batch_size", metavar="per iteration", type=int, default=4, choices=[4],
                        help="input the batch size (default: %(default)s)")
    parser.add_argument("-f", "--buffer_size", dest="buffer_size", metavar="buffer size", type=int, default=150,
                        help="input the buffer size (default: %(default)s)")

    args = parser.parse_args()

    print("--------------------------* train configuration *--------------------------")
    print("    train_set_address:               ", args.trainset_address)
    print("    the place to store the model:    ", args.model_add)
    print("    require offline label generator: ", args.offline_label_generator)
    print("    json file of ground truth:       ", args.ground_truth)
    print("    num_epochs:                      ", args.num_epochs)
    print("    batch_size:                      ", args.batch_size)
    print("    buffer_size:                     ", args.buffer_size)
    print("***************************************************************************")

    if args.offline_label_generator == 'yes' and args.ground_truth is None:
        raise ValueError("missing the argument, json file of ground truth.  *** check arg: -gt | --ground_truth ***")

    return args


def parse_cmd_testing_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", dest ="test_dataset", metavar="INPUT-PATH", default="sample_data/test_set/",
                        help="The path to the prediction data folder. (default: %(default)s)")
    parser.add_argument("-o", dest="result_address", metavar="OUTPUT-PATH", default="tmp_pred/",
                        help="The path to put the prediction result folder. (default: %(default)s)")
    parser.add_argument("-m", dest="model", metavar="MODEL-PATH", default="pre_trained_model/",
                        help="The path to the model storage. (default: %(default)s)")
    parser.add_argument("-d", dest="display", metavar="DISPLAY", default="no",
                        choices=["yes", "no"], help="whether to display prediction result or not. (default: %("
                                                    "default)s)")

    args = parser.parse_args()

    print("-----------------------* prediction configuration *-----------------------")
    print("    prediction_set_address:        ", args.test_dataset)
    print("    prediction_result_address:     ", args.result_address)
    print("    model_storing_address:         ", args.model)
    print("    to display result or not:      ", args.display)
    print("***************************************************************************")
    return args


if __name__ == '__main__':
    parse_cmd_training_args()
    # parse_cmd_testing_args()