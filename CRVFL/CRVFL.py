#coding=utf-8
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import time
import scipy.io as sio
from sklearn.linear_model import Ridge
from numpy import loadtxt, atleast_2d
import os
import glob
from skimage import io,transform
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from tensorflow.examples.tutorials.mnist import input_data


plt.switch_backend('agg')
##这里定义一些全局的变量
L_max =20
Tmax = 10
Lambdas = [0.001, 0.01,0.1,0.5,1,10]  # 这里的范围应该需要调整一下
Lambdas_len = np.size(Lambdas)
r =  [ 0.9, 0.99, 0.999,0.9999, 0.99999, 0.999999]
r_len =np.size(r)

# 开启Eager Execution
tfe.enable_eager_execution()
data = input_data.read_data_sets("data/MNIST_data/", one_hot=True)

batch_size = 128


def _constant(stddev, shape):
    dtype =tf.float32
    var = tf.constant(tf.truncated_normal(shape,stddev=stddev,dtype=dtype))
    return var
def conv2d(x,W,b,strides=1):
    x = tf.reshape(x,[-1,28,28,1])
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return x
def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k+1,k+1,1],strides=[1,k,k,1],padding='SAME')
def conv_pool(x,W,b,strides=1,k=2):
    temp = tf.nn.sigmoid(conv2d(x,W,b,strides))
    temp = maxpool2d(temp,k)
    return temp
test_acc_array = []
train_acc_array = []
with tf.device("/gpu:0"):
    numC = 8
    W0 = _constant(0.01,[5,5,1,numC])
    b0 = _constant(0.01,[numC])

    start = time.time()

    regr = Ridge(alpha=0.000001)
    images_batch,labels_batch = data.train.next_batch(5000)
    images_batch = tf.cast(images_batch,tf.float32).gpu(0)
    labels_batch = tf.cast(labels_batch,tf.float32).gpu(0)

    hidden = conv_pool(images_batch,W0,b0)
    hidden = tf.reshape(hidden,[-1,14*14*numC])#特征图展开

    regr.fit(hidden, labels_batch)#线性回归


    #测试
    hidden = conv_pool(data.test.images, W0, b0)
    hidden = tf.reshape(hidden, [-1, 14 * 14 *numC])
    logits = regr.predict(hidden)
    loss = tf.sqrt(tf.reduce_mean(tf.square(logits - data.test.labels)))  # 计算均方根

    temp = tf.equal(tf.argmax(logits, 1), tf.argmax(data.test.labels, 1))
    acc = tf.reduce_mean(tf.cast(temp, tf.float32))
    test_acc_array.append(acc.numpy())

    print("test loss = {}".format(loss.numpy()) + "  acc = {}".format(acc.numpy()))
