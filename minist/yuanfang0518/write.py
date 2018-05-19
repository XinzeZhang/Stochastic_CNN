import numpy as np
import tensorflow as tf
import scipy.misc
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST-data/", dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
# 生成整数的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def eval_train_data(batch_x):
    x = tf.reshape(batch_x, shape=[28, 28,1])
    res = tf.image.resize_image_with_crop_or_pad(x, 24, 24)
    res = tf.image.per_image_standardization(res)
    print(tf.shape(res))
    res = tf.reshape(res, [24, 24])
    return res
batch= tf.placeholder(tf.float32,[784]) # dropout (keep probability)

myimages = eval_train_data(batch)

train_examples = mnist.train.num_examples
test_examples = mnist.test.num_examples
# 存储TFRecord文件的地址
if os.path.exists('./record')==False:
    os.mkdir('./record')
train_filename = './record/output_train.tfrecords'
test_filename = './record/output_test.tfrecords'
# 创建一个writer来写TFRecord文件
train_writer = tf.python_io.TFRecordWriter(train_filename)
test_writer = tf.python_io.TFRecordWriter(test_filename)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
# 将每张图片都转为一个Example
def save_img(img1,img2,i,label,type="_train"):
    img1 = np.reshape(img1,[28,28])
    scipy.misc.imsave('./record/'+str(label)+type+str(i)+'_1.png',img1)
    scipy.misc.imsave('./record/'+str(label)+type+str(i)+'_2.png',img2)

for i in range(test_examples):
    # image_raw = images[i].tostring()  # 将图像转为字符串
    # image_raw = np.reshape(images[i],[28,28])
    # image_byte = image_raw.tobytes()
    image_argu = sess.run(myimages,feed_dict={batch:mnist.test.images[i]})

    image_byte = image_argu.tostring()
    # print(np.argmax(mnist.test.labels[i]))

    if(i%5000==0):
        print("step"+str(i))
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'label': _int64_feature(np.argmax(mnist.test.labels[i])),
            'image_raw': _bytes_feature(image_byte)
        }))
    test_writer.write(example.SerializeToString())  # 将Example写入TFRecord文件

print('data processing success')
test_writer.close()


for i in range(train_examples):
    # image_raw = images[i].tostring()  # 将图像转为字符串
    # image_raw = np.reshape(images[i],[28,28])
    # image_byte = image_raw.tobytes()
    image_argu = sess.run(myimages,feed_dict={batch:mnist.train.images[i]})
    image_byte = image_argu.tostring()
    # save_img(mnist.train.images[i],image_argu,i,np.argmax(mnist.train.labels[i]))
    # print(np.argmax(mnist.train.labels[i]))
    if(i%5000==0):
        print("step"+str(i))
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'label': _int64_feature(np.argmax(mnist.train.labels[i])),
            'image_raw': _bytes_feature(image_byte)
        }))
    train_writer.write(example.SerializeToString())  # 将Example写入TFRecord文件

print('data processing success')
train_writer.close()
