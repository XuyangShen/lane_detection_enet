# @author: Xinqi Zhu
# @Date: 2019-04-25
# @Editor: atom

# script for building an ENet
import enet.ENet_Components as ec
import tensorflow as tf


def inverse_class_weight(label_tensor):
    """
    get pixelwise weight for computing loss
    higher loss for false negatives

    @params
    label_tensor: (batch_size)4*512*512*2 where 0 channel is non-lane probs
    """
    none_lane_tensor = label_tensor[:, :, :, 0]
    lane_tensor = label_tensor[:, :, :, 1]

    # get probabilities
    y1 = tf.cast(tf.reduce_sum(none_lane_tensor, [1, 2]), dtype=tf.dtypes.float32)
    y2 = tf.cast(tf.reduce_sum(lane_tensor, [1, 2]), dtype=tf.dtypes.float32)
    y_sum = tf.add(y1, y2)
    ys = [tf.divide(y1, y_sum), tf.divide(y2, y_sum)]

    # compute weights: weight = 1.0/ln(c+prob)
    c = tf.constant(1.02)
    ys = tf.add(c, ys)
    ys = tf.divide(tf.constant(1.0), tf.log(ys))

    # repeat to get (batch_size)4*512*512*2 where each element
    # is weight on that pixel
    result_list = []
    for index in range(2):  # 0 channel and 1 channel
        layer_list = []

        weights = ys[index]
        for batch_index in range(weights.get_shape()[0]):
            # all batches(images)
            layer = tf.ones((512, 512))
            layer_list.append(tf.multiply(layer, weights[batch_index]))
        result_list.append(tf.stack(layer_list))

    result = tf.stack(result_list, axis=3)
    return result


def prediction_interrupt(probabilities, batch_size=4):
    """
    (batch_size)4*512*512*2 where 0 channel is non-lane probs
    """
    result = []
    for layer in range(batch_size):
        none_lane_tensor = probabilities[layer, :, :, 0]
        lane_tensor = probabilities[layer, :, :, 1]
        none_lane_tensor = tf.reshape(none_lane_tensor, [512 * 512, 1])
        lane_tensor = tf.reshape(lane_tensor, [512 * 512, 1])

        upper = lane_tensor >= none_lane_tensor
        upper = tf.cast(upper, tf.int64)

        a = tf.ones(
            shape=[512 * 512, 1],
            dtype=tf.int64
        )
        lower = tf.math.subtract(
            a, upper
        )
        convert = tf.concat([lower, upper], 1)
        convert = tf.reshape(convert, [1, 512, 512, 2])

        result.append(convert)
    result = tf.stack(result)
    result = tf.reshape(result, [4, 512, 512, 2])
    return result


def ENet(features, labels, mode):
    # print("x is ", features.get_shape().as_list())
    # set up the input layer
    if mode == tf.estimator.ModeKeys.PREDICT:
        inputs = tf.reshape(features, [1, 512, 512, 3])
    else:
        inputs = tf.reshape(features, [4, 512, 512, 3])
        labels = tf.reshape(labels, [4, 512, 512])

    # define regularization rate
    r_rate1 = 0.01
    r_rate2 = 0.1

    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
    else:
        is_training = False

    # ---------------------------- inital block ----------------------------
    # Not factored into function since it is used only once

    # Convolutional branch
    conv_branch = tf.layers.conv2d(inputs=inputs, filters=13, kernel_size=[3, 3], strides=(2, 2), padding='same')
    conv_branch = tf.layers.batch_normalization(inputs=conv_branch, training=is_training)
    conv_branch = tf.keras.layers.PReLU()(conv_branch)
    # Max pool branch
    pool_branch = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=(2, 2))
    # Concatenated: 13+3
    init_block_result = tf.concat([conv_branch, pool_branch], axis=3)

    # ---------------------------- Stage1:encoder ----------------------------
    bottleneck1_0, pooling_indices_1, inputs_shape_1 = ec.bottleneck(
        inputs=init_block_result, is_training=is_training,
        kernel_size=3, regularizer_rate=r_rate1, output_fmap=64,
        is_downsampling=True)
    bottleneck1_1 = ec.bottleneck(inputs=bottleneck1_0, is_training=is_training,
                                  kernel_size=3, regularizer_rate=r_rate1, output_fmap=64,
                                  is_regular_conV=True)
    bottleneck1_2 = ec.bottleneck(inputs=bottleneck1_1, is_training=is_training,
                                  kernel_size=3, regularizer_rate=r_rate1, output_fmap=64,
                                  is_regular_conV=True)
    bottleneck1_3 = ec.bottleneck(inputs=bottleneck1_2, is_training=is_training,
                                  kernel_size=3, regularizer_rate=r_rate1, output_fmap=64,
                                  is_regular_conV=True)
    bottleneck1_4 = ec.bottleneck(inputs=bottleneck1_1, is_training=is_training,
                                  kernel_size=3, regularizer_rate=r_rate1, output_fmap=64,
                                  is_regular_conV=True)

    # ---------------------------- Stage2,3:encoder ----------------------------
    # from bottleneck2.0 onwards, use the second regularizer_rate
    bottleneck2_0, pooling_indices_2, inputs_shape_2 = ec.bottleneck(
        inputs=bottleneck1_4, is_training=is_training,
        kernel_size=3, regularizer_rate=r_rate2, output_fmap=128,
        is_downsampling=True)

    for i in range(2):
        enet = ec.bottleneck(inputs=bottleneck2_0, is_training=is_training,
                             kernel_size=3, regularizer_rate=r_rate2, output_fmap=128,
                             is_regular_conV=True)
        enet = ec.bottleneck(inputs=enet, is_training=is_training,
                             kernel_size=3, regularizer_rate=r_rate2, output_fmap=128,
                             is_dilated=True, dilation_rate=2)
        enet = ec.bottleneck(inputs=enet, is_training=is_training,
                             kernel_size=5, regularizer_rate=r_rate2, output_fmap=128,
                             is_asymmetric=True)
        enet = ec.bottleneck(inputs=enet, is_training=is_training,
                             kernel_size=3, regularizer_rate=r_rate2, output_fmap=128,
                             is_dilated=True, dilation_rate=4)
        enet = ec.bottleneck(inputs=enet, is_training=is_training,
                             kernel_size=3, regularizer_rate=r_rate2, output_fmap=128,
                             is_regular_conV=True)
        enet = ec.bottleneck(inputs=enet, is_training=is_training,
                             kernel_size=3, regularizer_rate=r_rate2, output_fmap=128,
                             is_dilated=True, dilation_rate=8)
        enet = ec.bottleneck(inputs=enet, is_training=is_training,
                             kernel_size=5, regularizer_rate=r_rate2, output_fmap=128,
                             is_asymmetric=True)
        enet = ec.bottleneck(inputs=enet, is_training=is_training,
                             kernel_size=3, regularizer_rate=r_rate2, output_fmap=128,
                             is_dilated=True, dilation_rate=16)

    # ---------------------------- Stage4:decoder ----------------------------
    bottleneck4_0 = ec.bottleneck(inputs=enet, is_training=is_training,
                                  kernel_size=3, regularizer_rate=r_rate2, output_fmap=64,
                                  is_unpooling=True, output_shape=inputs_shape_2,
                                  maxpool_indicies=pooling_indices_2, scope='stage4')
    bottleneck4_1 = ec.bottleneck(inputs=bottleneck4_0, is_training=is_training,
                                  kernel_size=3, regularizer_rate=r_rate2, output_fmap=64,
                                  is_regular_conV=True)
    bottleneck4_2 = ec.bottleneck(inputs=bottleneck4_1, is_training=is_training,
                                  kernel_size=3, regularizer_rate=r_rate2, output_fmap=64,
                                  is_regular_conV=True)

    # ---------------------------- Stage5:decoder ----------------------------
    bottleneck5_0 = ec.bottleneck(inputs=bottleneck4_2, is_training=is_training,
                                  kernel_size=3, regularizer_rate=r_rate2, output_fmap=16,
                                  is_unpooling=True, output_shape=inputs_shape_1,
                                  maxpool_indicies=pooling_indices_1, scope='stage5')
    bottleneck5_1 = ec.bottleneck(inputs=bottleneck5_0, is_training=is_training,
                                  kernel_size=3, regularizer_rate=r_rate2, output_fmap=16,
                                  is_regular_conV=True)
    # check point: bottleneck5_1 has shape of (1, 256, 256, 16)

    # ---------------------------- full conV ----------------------------
    # softmax and softmax is used here, but sigmoid and binary cross entropy is another option
    logits = tf.layers.conv2d_transpose(inputs=bottleneck5_1, filters=2, kernel_size=(2, 2),
                                        strides=(2, 2))
    probabilities = tf.nn.softmax(logits)
    # probabilities = tf.reshape(tensor= probabilities, shape = [inputs_shape[0], inputs_shape[1], inputs_shape[2]])

    # ---------------------------- prediction ----------------------------
    predictions = {
        "classes": probabilities
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # ---------------------------- Calculate Loss----------------------------
    # set up label type
    # labels = tf.cast(labels, tf.int32)
    labels = tf.one_hot(labels, 2, axis=-1)
    # weights = labels * np.array([0.1, 10])
    # print("one_hot: ", labels)
    weights = inverse_class_weight(labels)
    weight = tf.reduce_sum(tf.multiply(labels, weights), 3)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels,
        logits=logits,
        weights=weight,
        label_smoothing=0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM)
    total_loss = tf.losses.get_total_loss()

    # ---------------------------- Calculate accuracy----------------------------
    _, cfm = tf.metrics.mean_iou(
        labels=labels,
        predictions=prediction_interrupt(probabilities),
        num_classes=2
    )
    ac = cfm[1, 1] / (cfm[1, 1] + cfm[0, 1] + cfm[1, 0])
    mean_iou_score = {
        'mean_iou': tf.identity(ac, 'accuracy')
    }

    # ---------------------------- Configure the Training Op----------------------------

    # decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)

    lr = tf.train.exponential_decay(
        learning_rate=5e-4,
        global_step=tf.train.get_global_step(),
        decay_steps=5000,
        decay_rate=0.3,
        staircase=True)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr,
            epsilon=1e-08
        )
        train_op = tf.contrib.slim.learning.create_train_op(
            total_loss,
            optimizer
        )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            predictions=mean_iou_score
        )

    # ---------------------------- Configure the log----------------------------
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('accuracy', ac)
    tf.summary.merge_all()

    # # ---------------------------- evaluation ----------------------------
    eval_metrixcs_ops = {
        'accuracy': ac
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=eval_metrixcs_ops
        )
