"""Code for training DDFSeg."""
#Peichenhao
import tensorflow as tf
# tensorflow_addons
# import tensorflow._api.v2.compat.v1 as tf


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.compat.v1.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(x):
    epsilon = 1e-5
    mean, var = tf.nn.moments(x, [1, 2], keepdims=True)
    scale = tf.Variable(tf.ones([x.shape[-1]]))
    offset = tf.Variable(tf.zeros([x.shape[-1]]))
    normalized = (x - mean) / ((var + epsilon) ** 0.5)
    out = scale * normalized + offset
    return out


def batch_norm(x, is_training = True):
    out = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, center=True)(x, training=is_training)
    return out


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.01,
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True, keep_rate=None,
                   relufactor=0, norm_type=None, is_training=True):
    with  tf.compat.v1.variable_scope(name):

        conv = tf.compat.v1.layers.conv2d(
            inputconv, o_d, f_w, s_w, padding,
            activation=None,
            kernel_initializer= tf.compat.v1.truncated_normal_initializer(
                stddev=stddev
            ),
            bias_initializer=None
        )
        if not keep_rate is None:
            conv = tf.nn.dropout(conv, keep_rate)

        if do_norm:
            if norm_type is None:
                # print ("normalization type is not specified!")
                quit()
            elif norm_type=='Ins':
                  conv = instance_norm(conv)
            elif norm_type=='Batch':
                  conv = batch_norm(conv, is_training)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_conv2d_ga(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True, keep_rate=None,
                   relufactor=0, norm_type=None, is_training=True):
    with  tf.compat.v1.variable_scope(name):

        conv = tf.compat.v1.layers.conv2d(
            inputconv, o_d, f_w, s_w, padding,
            activation=None,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                stddev=stddev
            ),
           bias_initializer=tf.constant_initializer(0.0)
        )
        if not keep_rate is None:
            conv = tf.nn.dropout(conv, keep_rate)

        if do_norm:
            if norm_type is None:
                # print( "normalization type is not specified!")
                quit()
            elif norm_type=='Ins':
                conv = instance_norm(conv)
            elif norm_type=='Batch':
                conv = batch_norm(conv, is_training)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def dilate_conv2d(inputconv, i_d=64, o_d=64, f_h=7, f_w=7, rate=2, stddev=0.01,
                   padding="VALID", name="dilate_conv2d", do_norm=True, do_relu=True, keep_rate=None,
                   relufactor=0, norm_type=None, is_training=True):
    with tf.compat.v1.variable_scope(name):
        f_1 = tf.compat.v1.get_variable('weights', [f_h, f_w, i_d, o_d], initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
       # b_1 = tf.compat.v1.get_variable('biases', [o_d], initializer=tf.constant_initializer(0.0, tf.float32))
        di_conv_2d = tf.nn.atrous_conv2d(inputconv, f_1, rate=rate, padding=padding)

        if not keep_rate is None:
            di_conv_2d = tf.nn.dropout(di_conv_2d, keep_rate)

        if do_norm:
            if norm_type is None:
                # print ("normalization type is not specified!")
                quit()
            elif norm_type=='Ins':
                di_conv_2d = instance_norm(di_conv_2d)
            elif norm_type=='Batch':
                di_conv_2d = batch_norm(di_conv_2d, is_training)

        if do_relu:
            if(relufactor == 0):
                di_conv_2d = tf.nn.relu(di_conv_2d, "relu")
            else:
                di_conv_2d = lrelu(di_conv_2d, relufactor, "lrelu")

        return di_conv_2d


def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relufactor=0, norm_type=None, is_training=True):
    with tf.compat.v1.variable_scope(name):

        conv = tf.compat.v1.layers.conv2d_transpose(
            inputconv, o_d, [f_h, f_w],
            [s_h, s_w], padding,
            activation=None,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev),
            bias_initializer=tf.constant_initializer(0.0)
        )

        if do_norm:
            if norm_type is None:
                # print( "normalization type is not specified!")
                quit()
            elif norm_type=='Ins':
                conv = instance_norm(conv)
            elif norm_type=='Batch':
                conv = batch_norm(conv, is_training)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv
