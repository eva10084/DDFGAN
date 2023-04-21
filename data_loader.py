"""Code for training DDFSeg."""
#Peichenhao
#import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
import numpy as np

BATCH_SIZE = 2

def _decode_samples(image_list, shuffle=False):
#    这段代码定义了一个字典，decomp_feature，其中包含了八个tf.io.FixedLenFeature()方法创建的特征。这些特征将在Tensorflow中用于解析输入数据，其中：
#    dsize_dim0、dsize_dim1和dsize_dim2分别表示原始数据的尺寸。
#    lsize_dim0、lsize_dim1和lsize_dim2分别表示标签数据的尺寸。
#    data_vol和label_vol分别是原始数据和标签数据，两者都是二进制格式的字符串。
    decomp_feature = {
        'dsize_dim0': tf.io.FixedLenFeature([], tf.int64),
        'dsize_dim1': tf.io.FixedLenFeature([], tf.int64),
        'dsize_dim2': tf.io.FixedLenFeature([], tf.int64),
        'lsize_dim0': tf.io.FixedLenFeature([], tf.int64),
        'lsize_dim1': tf.io.FixedLenFeature([], tf.int64),
        'lsize_dim2': tf.io.FixedLenFeature([], tf.int64),
        'data_vol': tf.io.FixedLenFeature([], tf.string),
        'label_vol': tf.io.FixedLenFeature([], tf.string)}

# 图片格式
    raw_size = [512, 512, 1]   # 192
    volume_size = [512, 512, 1]  # 192
    label_size = [512, 512, 1]   # 192

# 数据预处理的起点，将数据读取到内存中供后续处理使用。
# 在数据处理过程中，通常需要将原始数据转换为模型能够接受的形式，如将图像数据解码为 Tensor 数组、进行图像增强等操作。
# 这些操作通常是由 Tensorflow 的 tf.data.Dataset 类实现的。
    data_queue = tf.train.string_input_producer(image_list, shuffle=shuffle)
# 第一个参数 image_list 是一个字符串类型的列表，表示所有数据的路径列表；第二个参数 shuffle 表示是否随机打乱数据的读取顺序。
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(data_queue)
# 在 data_queue 中读取一个文件路径并返回一个字符串类型的张量 serialized_example
# 然后使用 TFRecordReader 对象的 read 方法将其解析为一个张量对 (fid, serialized_example)，其中 fid 是读取的文件的名称。
    parser = tf.parse_single_example(serialized_example, features=decomp_feature) # 上述字典
# 将 serialized_example 解析为一个 Python 字典，其中包含了对应的 Tensor 对象。
# 这个字典的键对应于 decomp_feature 字典的键，值则是一个解析后的 Tensor 对象。
# 通过这个函数，我们可以将二进制字符串类型的 TFRecord 数据解析为 Tensorflow 的 Example 类型的对象，以供后续处理。

# 这段代码是对解析后的 TFRecord 数据进行进一步的处理。
# 使用 tf.decode_raw 函数将解析后的 parser['data_vol'] 字段解码为一个 float32 类型的 Tensor 对象 data_vol。
    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, raw_size) # 将 data_vol 的形状调整为 raw_size，
# 从 data_vol 中截取大小为 volume_size 的子张量，并将其赋值给 data_vol 变量。
# 通常用于将原始的图像数据剪裁为指定的大小，以满足神经网络的输入大小要求。
    data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)

    label_vol = tf.decode_raw(parser['label_vol'], tf.float32)
    label_vol = tf.reshape(label_vol, raw_size)
    label_vol = tf.slice(label_vol, [0, 0, 0], label_size)

    batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 4)
# 它首先使用 tf.squeeze 函数将 label_vol 张量的所有尺寸为1的维度都删除掉，得到一个形状为 [batch_size] 的张量。
# 然后使用 tf.cast 函数将 label_vol 转换为 uint8 类型的张量，以便后面的 tf.one_hot 函数使用。
# 最后，使用 tf.one_hot 函数将 uint8 类型的 label_vol 张量转换为 one-hot 编码的形式。
# tf.one_hot 函数需要两个参数：一个是待编码的整数张量，另一个是编码后的维度。这里将 label_vol 张量转换为 one-hot 编码后的维度为 4，因此 tf.one_hot 函数的第二个参数是 4。

    return tf.expand_dims(data_vol[:, :, 0], axis=2), batch_y
# 其中 a 是输入数据经过处理后的结果，是一个形状为 (height, width, 1) 的三维张量，表示一个二维图像。
# 具体来说，这行代码使用 TensorFlow 的 tf.expand_dims 函数在最后一个维度上增加一个长度为1的新维度，将 data_vol[:, :, 0] 张量从二维转换为三维。
# 其中，data_vol[:, :, 0] 取出了输入数据的第一层，并转换为一个二维张量。
# b 是一个 one-hot 编码形式的标签，是一个形状为 (4,) 的一维张量。
# 具体来说，这个标签使用 TensorFlow 的 batch_y 张量来表示，其中 batch_y 张量的每个元素是一个长度为 4 的 one-hot 编码，用来表示输入图像的类别。

def _load_samples(source_pth, target_pth):
    # 将txt中存储的文件名称路径依次赋值
    # a源域，b目标域
    with open(source_pth, 'r') as fp:
        rows = fp.readlines()    # 读取txt文件中的每一行
    imagea_list = [row[:-1] for row in rows]    # 每张图片路径的list

    # 这段代码的作用是将指定路径（source_pth）下的txt的每一行读取出来，并将其存储为一个列表（imagea_list）
    # 列表中的每个元素表示文件中的一行内容，并且去掉每行的末尾换行符。

    with open(target_pth, 'r') as fp:
        rows = fp.readlines()
    imageb_list = [row[:-1] for row in rows]

    # print(imagea_list)
    data_vola, label_vola = _decode_samples(imagea_list, shuffle=True)  # 调用上级
    data_volb, label_volb = _decode_samples(imageb_list, shuffle=True)

    return data_vola, data_volb, label_vola, label_volb


def load_data(source_pth, target_pth, do_shuffle=True):

    # image_i, image_j,源域和目标域的图像；  gt_i, gt_j  源域和目标域的标注
    image_i, image_j, gt_i, gt_j = _load_samples(source_pth, target_pth)  # 从txt中读取文件名，并打开对应图片，调用上级
    # (512, 512, 1)(512, 512, 1)(512, 512, 4)(512, 512, 4)

    # print(image_i.shape)
    # 在第三个维度上将这三个相同的张量拼接在一起，形成了一个新的形状为 (height, width, 3) 的三维张量。
    # 这样做的目的是将单通道的灰度图转换为三通道的 RGB 彩色图，方便在可视化时显示。
    image_i = tf.concat((image_i,image_i,image_i), axis=2)
    image_j = tf.concat((image_j,image_j,image_j), axis=2)

    # 三通道(512, 512, 3)(512, 512, 3)(512, 512, 4)(512, 512, 4)
    # print(image_i.shape)
    # print(image_j.shape)
    # print(gt_i.shape)
    # print(gt_j.shape)

    # Batch
    if do_shuffle is True:
        # tf.train.shuffle_batch 函数用于随机批处理，
        # tf.train.shuffle_batch 会启动多个线程异步地读取数据，并打乱输入数据的顺序，使得每个批次中的样本都具有一定的随机性。
        images_i, images_j, gt_i, gt_j = tf.train.shuffle_batch([image_i, image_j, gt_i, gt_j], BATCH_SIZE, 10, 5)
    else:
        # 顺序批处理，其参数与 tf.train.shuffle_batch 相同，但不会对输入数据的顺序进行打乱
        images_i, images_j, gt_i, gt_j = tf.train.batch([image_i, image_j, gt_i, gt_j], batch_size=BATCH_SIZE, num_threads=1, capacity=100)

    return images_i, images_j, gt_i, gt_j

def _decode_test_samples(image_list, shuffle=False):
    decomp_feature = {
        'dsize_dim0': tf.FixedLenFeature([], tf.int64),
        'dsize_dim1': tf.FixedLenFeature([], tf.int64),
        'dsize_dim2': tf.FixedLenFeature([], tf.int64),
        'lsize_dim0': tf.FixedLenFeature([], tf.int64),
        'lsize_dim1': tf.FixedLenFeature([], tf.int64),
        'lsize_dim2': tf.FixedLenFeature([], tf.int64),
        'data_vol': tf.FixedLenFeature([], tf.string),
        'label_vol': tf.FixedLenFeature([], tf.string)}

    raw_size = [128, 128, 1]
    volume_size = [128, 128, 1]

    data_queue = tf.train.string_input_producer(image_list, shuffle=False)
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(data_queue)
    parser = tf.parse_single_example(serialized_example, features=decomp_feature)

    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, raw_size)
    data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)

    return tf.expand_dims(data_vol[:, :, 0], axis=2)


def _load_test_samples(source_pth):

    with open(source_pth, 'r') as fp:
        rows = fp.readlines()
    imagea_list = [row[:-1] for row in rows]

    data_vola= _decode_test_samples(imagea_list, shuffle=False)

    return data_vola


def load_testdata(source_pth, do_shuffle=False):

    image_i= _load_test_samples(source_pth)
    # print(image_i.shape)

    image_i = tf.concat((image_i,image_i,image_i), axis=2)
 # print(image_i.shape)

    # Batch
    if do_shuffle is True:
        images_i= tf.train.shuffle_batch([image_i], BATCH_SIZE, 500, 100)
    else:
        images_i = tf.train.batch([image_i], batch_size=BATCH_SIZE, num_threads=1, capacity=500)

    return images_i
