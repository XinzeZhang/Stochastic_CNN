import tensorflow as tf
import numpy as np
import scipy.misc
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

# 创建一个reader来读取TFRecord文件中的Example
reader = tf.TFRecordReader()
mnist = input_data.read_data_sets("MNIST-data/", one_hot=True)

# 创建一个队列来维护输入文件列表
filename_queue_test = tf.train.string_input_producer(['./record/output_test.tfrecords'])
filename_queue_train =  tf.train.string_input_producer(['./record/output_train.tfrecords'])
# 从文件中读出一个Example
_, serialized_example_test = reader.read(filename_queue_test)
_, serialized_example_train = reader.read(filename_queue_train)

# 用FixedLenFeature将读入的Example解析成tensor
features_test = tf.parse_single_example(
    serialized_example_test,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
features_train = tf.parse_single_example(
    serialized_example_train,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

# 将字符串解析成图像对应的像素数组
images_test = tf.decode_raw(features_test['image_raw'], tf.float32)
labels_test = tf.cast(features_test['label'], tf.int32)
images_train = tf.decode_raw(features_train['image_raw'], tf.float32)
labels_train = tf.cast(features_train['label'], tf.int32)
sess = tf.Session()

# 启动多线程处理输入数据
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

res_test_images=[]
res_test_labels =[]
res_train_images=[]
res_train_labels =[]

def test_data():
    # 每次运行读取一个Example。当所有样例读取完之后，在此样例中程序会重头读取
    for i in range(10000):
        image, label = sess.run([images_test, labels_test])
        # print(image)
        img = np.reshape(image, [24, 24, 1])
        res_test_images.append(img)
        lb = np.zeros([10])
        lb[label] = 1
        res_test_labels.append(lb)

    res_image = np.array(res_test_images)
    res_image = np.reshape(res_image,[10000,24,24,1])

    res_label = np.array(res_test_labels)

    return res_image,res_label

def train_data():
    for i in range(55000):
        image, label = sess.run([images_train, labels_train])
        img = np.reshape(image, [24, 24])
        res_train_images.append(img)

        # print(img)
        lb = np.zeros([10])
        lb[label] = 1

        res_train_labels.append(lb)
    res_image = np.array(res_train_images)
    res_image = np.reshape(res_image,[55000,24,24,1])
    res_label = np.array(res_train_labels)

    return res_image,res_label

