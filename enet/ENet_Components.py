#@author: Xinqi Zhu
#@Date: 2019-04-25
#@Editor: atom

#@Assumptions:
# during training, train data is 512*512*3 image,
#                  label is 512*512*2 (2 classes: lane pixel or not)

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


def unpool(inputs, down_pool_indices, output_shape, k_size = [1,2,2,1]):
    """
    unpooling: pad with 0 for all indices that are not among
    agrmax from previous down pool

    @params
    inputs: 4D tensor
    down_pool_indices: 4D tensor of argmax from previous down pool
    output_shape: list

    @return
    tensor after unpool layer
    """

    down_pool_indices = tf.cast(down_pool_indices, tf.int32)
    input_shape = tf.cast(inputs.get_shape(), tf.int32)

    ones_mask = tf.ones_like(down_pool_indices, dtype = tf.int32)  # set all entries to 1
    b_shape = tf.convert_to_tensor([input_shape[0],1,1,1])
    b_range = tf.reshape(tf.range(output_shape[0], dtype = tf.int32), shape = b_shape)
    # compute indices for different dimensions
    dim_batch = tf.multiply(ones_mask, b_range)
    dim_height = down_pool_indices // (output_shape[2] * output_shape[3])
    dim_width = (down_pool_indices // output_shape[3]) % output_shape[2]
    dim_feature = tf.multiply(ones_mask, tf.range(output_shape[3], dtype= tf.int32))

    inputs_element_No = tf.size(inputs)
    indices = tf.transpose(tf.reshape(tf.stack([dim_batch, dim_height, dim_width, dim_feature]), [4, inputs_element_No]))
    values = tf.reshape(inputs, [inputs_element_No])

    return tf.scatter_nd(indices, values, output_shape)


def bottleneck(inputs, is_training, kernel_size, regularizer_rate, output_fmap,
               dilation_rate = None, maxpool_indicies=None, output_shape=None,
               is_regular_conV=False, is_downsampling=False, is_dilated=False,
               is_asymmetric=False, is_unpooling=False,
               scope=''):
    """
    @params
    inputs: 4D tensor
    is_training: boolean indicating if is training phase
    kernel_size: integer
    regularizer_rate: regularization probability
    output_fmap: required number of feature map for output
    dilation_rate: only required if is_dilated is True
    maxpool_indicies: only required if is_unpooling is True
    output_shape: only required if is_unpooling is True

    is_regular_conV: boolean
    is_downsampling: boolean
    is_dilated:      boolean
    is_asymmetric:   boolean
    is_unpooling:    boolean

    scope: string

    @return
    new 4D tensor
    """
    # during 1*1 projection, number of feature map would be shrinked to 1/4
    projection_rate = 4;
    projected_fmap = int((inputs.get_shape().as_list())[3] / projection_rate);
    # expand to the required number of feature map
    expanded_fmap =  output_fmap

    # ----------------regular conV in extension branch----------------
    # NOTE: input and output shape have exactly same shape
    if is_regular_conV:
        # main branch
        main_branch = inputs # simple copy

        # extension branch
        # 1*1 projection
        ext_branch = tf.layers.conv2d(inputs = inputs, filters = projected_fmap, kernel_size = (1,1), strides = (1,1), padding = 'valid')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # regular convolution: padding set to same to keep width, length unchanged
        ext_branch = tf.layers.conv2d(inputs = ext_branch, filters = projected_fmap, kernel_size = (kernel_size, kernel_size), strides = (1,1), padding = 'same')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # 1*1 expansion
        ext_branch = tf.layers.conv2d(inputs = ext_branch, filters = expanded_fmap, kernel_size = (1, 1), strides = (1,1), padding = 'valid')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # regularizer: spatial dropout
        ext_branch = tf.keras.layers.SpatialDropout2D(rate = regularizer_rate)(ext_branch)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)

        # element wise add with main_branch and extension branch
        bottlenecked = tf.add(main_branch, ext_branch)
        bottlenecked = tf.keras.layers.PReLU()(bottlenecked)

        return bottlenecked

    # ----------------downsampling in extension branch----------------
    # NOTE: output shape half the width and length, but increase the number of feature map
    # e.g. 16*256*256
    #  --> 64*128*128
    elif is_downsampling:
        # main branch
        # maxpooling with the max indices also returned, for later unpooling
        main_branch, max_indices = tf.nn.max_pool_with_argmax(input = inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding= 'SAME')

        input_shape = inputs.get_shape().as_list()
        fmap_diff = output_fmap - input_shape[3]
        # pads the difference with 0s on the fmap axis after real contents
        pads = [[0,0],[0,0],[0,0],[0, fmap_diff]]
        pads_tensor = tf.convert_to_tensor(pads)
        main_branch = tf.pad(main_branch, paddings= pads_tensor)

        # extension branch
        # 2*2 conv with stride2 to replace the normal 1*1 projection
        ext_branch = tf.layers.conv2d(inputs = inputs, filters = projected_fmap, kernel_size = (2,2), strides = (2,2), padding = 'same')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # regular convolution: padding set to same to keep width, length unchanged
        ext_branch = tf.layers.conv2d(inputs = ext_branch, filters = projected_fmap, kernel_size = (kernel_size, kernel_size), strides = (1,1), padding = 'same')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # 1*1 expansion
        ext_branch = tf.layers.conv2d(inputs = ext_branch, filters = expanded_fmap, kernel_size = (1, 1), strides = (1,1), padding = 'valid')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # regularizer: spatial dropout
        ext_branch = tf.keras.layers.SpatialDropout2D(rate = regularizer_rate)(ext_branch)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)

        # element wise add with main_branch and extension branch
        bottlenecked = tf.add(main_branch, ext_branch)
        bottlenecked = tf.keras.layers.PReLU()(bottlenecked)

        # inputs shape here is the expected output shape for corresponding unpooling
        return bottlenecked, max_indices, input_shape

    # ----------------dilated conV in extension branch----------------
    # NOTE: input and output shape have exactly same shape, also the dilation_rate must be specified
    # dilated conV can be used to increase recpetion field
    elif is_dilated:
        if dilation_rate is None:
            raise ValueError("The dilation rate must be set!")
        # main branch
        main_branch = inputs # simple copy

        # extension branch
        # 1*1 projection
        ext_branch = tf.layers.conv2d(inputs = inputs, filters = projected_fmap, kernel_size = (1,1), strides = (1,1), padding = 'valid')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # dilated convolution
        ext_branch = tf.layers.conv2d(inputs = ext_branch, filters = projected_fmap, kernel_size = (kernel_size, kernel_size),
                                      strides = (1,1), padding = 'same', dilation_rate=dilation_rate)
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # 1*1 expansion
        ext_branch = tf.layers.conv2d(inputs = ext_branch, filters = expanded_fmap, kernel_size = (1, 1), strides = (1,1), padding = 'valid')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # regularizer: spatial dropout
        ext_branch = tf.keras.layers.SpatialDropout2D(rate = regularizer_rate)(ext_branch)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)

        # element wise add with main_branch and extension branch
        bottlenecked = tf.add(main_branch, ext_branch)
        bottlenecked = tf.keras.layers.PReLU()(bottlenecked)

        return bottlenecked

    # ----------------asymmetric conV in extension branch----------------
    # NOTE: input and output shape have exactly same shape
    elif is_asymmetric:
        # main branch
        main_branch = inputs # simple copy

        # extension branch
        # 1*1 projection
        ext_branch = tf.layers.conv2d(inputs = inputs, filters = projected_fmap, kernel_size = (1,1), strides = (1,1), padding = 'valid')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # asymmetric convolution: break conV into 2 sub conV
        ext_branch = tf.layers.conv2d(inputs = ext_branch, filters = projected_fmap, kernel_size = (kernel_size, 1), strides = (1,1), padding = 'same')
        ext_branch = tf.layers.conv2d(inputs = ext_branch, filters = projected_fmap, kernel_size = (1, kernel_size), strides = (1,1), padding = 'same')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # 1*1 expansion
        ext_branch = tf.layers.conv2d(inputs = ext_branch, filters = expanded_fmap, kernel_size = (1, 1), strides = (1,1), padding = 'valid')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # regularizer: spatial dropout
        ext_branch = tf.keras.layers.SpatialDropout2D(rate = regularizer_rate)(ext_branch)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)

        # element wise add with main_branch and extension branch
        bottlenecked = tf.add(main_branch, ext_branch)
        bottlenecked = tf.keras.layers.PReLU()(bottlenecked)

        return bottlenecked

    # ----------------unpooling in main branch----------------
    # NOTE: output shape doubles the width and length, but reduces the number of feature map
    # e.g. 128*64*64
    #  --> 64*128*128
    elif is_unpooling:
        # need to specify both output_shape and maxpool_indicies
        if output_shape is None:
            raise ValueError("Must specify the output shape!")
        if maxpool_indicies is None:
            raise ValueError("Must specify the indicies during former maxpooling!")
        # main branch
        # unpooling is in main branch
        main_branch = tf.layers.conv2d(inputs = inputs, filters = output_fmap,
                                       kernel_size = (1,1), strides = (1,1), padding = 'valid') # reduce fmap
        main_branch = tf.layers.batch_normalization(inputs = main_branch, training = is_training)
        main_branch = unpool(inputs= main_branch, down_pool_indices=maxpool_indicies, k_size=[1, 2, 2, 1], output_shape=output_shape)

        # extension branch
        # 1*1 projection
        ext_branch = tf.layers.conv2d(inputs = inputs, filters = projected_fmap, kernel_size = (1,1), strides = (1,1), padding = 'valid')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # a deconV to match the shape(width and length)
        # e.g. 128*64*64 -> 128/4*64*64(after projection) -> 128/4*128*128(pad with 0)
        main_branch_shape = main_branch.get_shape().as_list()
        intermedian_shape = tf.convert_to_tensor([main_branch_shape[0], main_branch_shape[1], main_branch_shape[2], projected_fmap])
        filter_size = [kernel_size, kernel_size, projected_fmap, projected_fmap]
        filters = tf.get_variable(name=scope+'_alpha',shape=filter_size, initializer=initializers.xavier_initializer(),
                                  dtype=tf.float32)
        ext_branch = tf.nn.conv2d_transpose(value=ext_branch, filter=filters, strides=[1, 2, 2, 1],
                                            output_shape=intermedian_shape)
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # 1*1 expansion
        ext_branch = tf.layers.conv2d(inputs = ext_branch, filters = expanded_fmap, kernel_size = (1, 1), strides = (1,1), padding = 'valid')
        ext_branch = tf.layers.batch_normalization(inputs = ext_branch, training = is_training)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)
        # regularizer: spatial dropout
        ext_branch = tf.keras.layers.SpatialDropout2D(rate = regularizer_rate)(ext_branch)
        ext_branch = tf.keras.layers.PReLU()(ext_branch)

        # element wise add with main_branch and extension branch
        bottlenecked = tf.add(main_branch, ext_branch)
        bottlenecked = tf.keras.layers.PReLU()(bottlenecked)

        return bottlenecked
