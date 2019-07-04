# -*- coding: utf-8 -*-
# @Time    : May 25
# @Author  : Xuyang SHEN, Alisdair Cameron, Xinqi Zhu
# @File    : training.py
# @IDE: PyCharm Community Edition

import tensorflow as tf
import time

from enet.ENet import *
from data_provider.data_dataset import *
from data_provider.label2tfrecord import *
from config import *

# ---------------------------------------------------------
# run parser
# ---------------------------------------------------------
config = parse_cmd_training_args()

# import config from parser
train_add = config.trainset_address
model_add = config.model_add
epochs = config.num_epochs
batch_size = config.batch_size
buffer_size = config.buffer_size

# ---------------------------------------------------------
# offline label generator
# ---------------------------------------------------------
if config.offline_label_generator == 'yes':
    print("---------------------* offline label generator begins *--------------------")
    tusimple = TfGenerator(
        path=train_add,
        js_name=config.ground_truth
    ).run()
    print("***************************************************************************")

print("program begins to training, at", time.ctime(),'\n')
starter = time.time()
# ---------------------------------------------------------
# initial data
# ---------------------------------------------------------
data = TDataset(
    json_add=train_add,
    num_epochs=epochs,
    buffer_size=buffer_size
)

print("\n")
# ---------------------------------------------------------
# tainning logging information
# ---------------------------------------------------------
tf.logging.set_verbosity(tf.logging.INFO)
tensors_to_log = {'mean_iou': 'accuracy'}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

est_config = tf.estimator.RunConfig(
    save_checkpoints_secs=20 * 60,  # times(second) 60s/step | 9-10 steps/epochs
    keep_checkpoint_max=40,
)

# ---------------------------------------------------------
# initialized the model
# ---------------------------------------------------------
model_address = os.path.join(os.getcwd(), model_add)
lane_detect = tf.estimator.Estimator(
    model_fn=ENet,
    model_dir=model_address,
    config=est_config
)

print("---------------------********** training **********s--------------------")
# ---------------------------------------------------------
# training process

lane_detect.train(
    input_fn=data.dataset_input_fn,
    steps=None,
    hooks=[logging_hook]
)

print("All the training is finished", " It takes ",
      "%0.2f" % (time.time() - starter), 'seconds')
print("\ntraining model store at: ", model_address)
