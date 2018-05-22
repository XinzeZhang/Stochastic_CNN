from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
import math
import os
import read
import sys
import scipy.misc
import seaborn as sns
# Import MNIST data
numberOfnet = sys.argv[1]
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST-data/", one_hot=True)
print(np.size(mnist.train.labels))
learning_rate = 0.0001
batch_size = 64
train_size = np.shape(mnist.train.images)[0]
display_step = int(math.ceil(train_size/batch_size))
num_steps = display_step*200
print(display_step)
allTrain =True
os.environ['CUDA_VISIBLE_DEVICES']='0'
if os.path.exists("./Argu_Result_npz") == False:
    os.mkdir("./Argu_Result_npz")
# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, 24,24,1])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

train_images,train_labels = read.train_data()
test_images,test_labels = read.test_data()

def save_img(img1,i,label,type="train"):
    if os.path.exists('./train_image/'+str(label)+"/") == False:
        os.mkdir('./train_image/'+str(label)+"/")
    scipy.misc.imsave('./train_image/'+str(label)+"/"+type+str(i)+'_1.png',img1)
#
# for i in range(55000):
#     temp = np.reshape(train_images[i], [24, 24])
#     save_img(temp, i, np.argmax(train_labels[i]))

def train_data_argument_batch(batch_x):
    x = tf.reshape(batch_x, shape=[-1, 28, 28,1])
    res = tf.random_crop(x[0], [24, 24, 1])
    # res = tf.image.random_flip_left_right(res)
    res = tf.image.per_image_standardization(res)
    res = tf.reshape(res, [1, 24, 24, 1])
    for i in range(1,batch_size):
        distorted_image = tf.random_crop(x[i], [24, 24, 1])
        # distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.per_image_standardization(distorted_image)
        distorted_image = tf.reshape(distorted_image, [1, 24, 24, 1])
        res = tf.concat([res,distorted_image],0)
    return res
raw_image = tf.placeholder(tf.float32,[None,784])
train_argu =train_data_argument_batch(raw_image)
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x= tf.nn.batch_normalization(x,0.001,1.0,0,1,0.0001)

    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    # x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
def create_graph():
    # Store layers weight & bias
    numC1 = 32  # the kernel num of the first convolution layer
    numC2 = 64 #the kernel num of the second convolution layer
    num_fc1 =1024 #the hidden num of the first full-connected layer
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(np.array(np.random.normal(0,0.1,[5,5,1,numC1]),np.float32),trainable=allTrain),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(np.array(np.random.normal(0,0.1,[5,5,numC1,numC2]),np.float32),trainable=allTrain),
        # # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(np.array(np.random.normal(0,0.1,[6*6*numC2,num_fc1]),np.float32),trainable=allTrain),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(np.array(np.random.normal(0,0.1,[num_fc1,num_classes]),np.float32))
    }

    biases = {
        'bc1': tf.Variable(tf.constant(0.1,shape=[numC1]),trainable=allTrain,dtype=np.float32),
        'bc2': tf.Variable(tf.constant(0.1,shape=[numC2]),trainable=allTrain,dtype=np.float32),
        'bd1': tf.Variable(tf.constant(0.1,shape=[num_fc1]),trainable=allTrain,dtype=np.float32),
        'out': tf.Variable(tf.constant(0.1,shape=[num_classes]),dtype=np.float32)
    }

    # Construct model
    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer


    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return weights,biases,prediction,loss_op,train_op,accuracy


weights,biases,prediction,loss_op,train_op,accuracy = create_graph() #create graph

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
test_loss_array = []
train_loss_array = []
test_acc_array = []
train_acc_array = []
prediction_array = []
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    start = time.clock()
    e_loss=0
    e_acc =0

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        data_argu = sess.run(train_argu,feed_dict={raw_image:batch_x})

        # print(np.shape(data_argu[1]))
        # Run optimization op (backprop)

        _,t_loss,t_acc = sess.run([train_op,loss_op,accuracy], feed_dict={X: data_argu, Y: batch_y, keep_prob: 0.5})
        # print("step"+str(step),t_loss,t_acc)
        if step % display_step == 0 :
            # Calculate batch loss and accuracy
            train_loss =0
            train_acc =0

            for i in range(100):  # calculate the train acc and loss
                start_index = i * 550
                end_index = (i + 1) * 550
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_images[start_index:end_index],
                                                                     Y: train_labels[start_index:end_index],
                                                                     keep_prob: 1.0})
                # print(loss,acc)
                train_loss +=loss
                train_acc +=acc

            train_loss_array.append(train_loss/100)
            train_acc_array.append(train_acc/100)
            print("Epoch " + str(int(step/display_step)) + ", Minibatch Loss= " + \
                  "{:.4f}".format(train_loss/100) + ", Training Accuracy= " + \
                  "{:.5f}".format(train_acc/100))

            test_acc=0
            test_loss=0
            for i in range(10):# calculate the test acc and loss
                start_index = i * 1000
                end_index = (i + 1) * 1000
                acc,loss= sess.run([accuracy,loss_op], feed_dict={X: test_images[start_index:end_index],
                                                     Y: test_labels[start_index:end_index],
                                                     keep_prob: 1.0})
                test_loss+=loss
                test_acc+=acc
            test_loss_array.append(test_loss / 10)
            test_acc_array.append(test_acc / 10)
            print("Epoch " + str(int(step/display_step)) + ", Minibatch Loss= " + \
                  "{:.4f}".format(test_loss/10) + ", test Accuracy= " + \
                  "{:.5f}".format(test_acc/10))
    print("cost_time: %.6f"% (time.clock() - start))
    print("Optimization Finished!")

    acc = 0
    for i in range(10):
        start_index = i * 1000
        end_index = (i + 1) * 1000
        temp_acc ,res_prediction= sess.run([accuracy,prediction] ,feed_dict={X: test_images[start_index:end_index],
                                             Y: test_labels[start_index:end_index],
                                             keep_prob: 1.0})
        acc+=temp_acc
        prediction_array.append(np.argmax(res_prediction,1))

    print('%.6f' % (acc / 10))
    temp_weights, temp_biases = sess.run([weights, biases])


    if allTrain:
        np.savez("./Argu_Result_npz/acc0_"+numberOfnet+".npz", test_acc_array, train_acc_array)
        # np.savez("./Argu_Result_npz/loss0.npz", test_loss_array, train_loss_array)
        np.savez("./Argu_Result_npz/pred0_"+numberOfnet+".npz",prediction_array)

    else:
        np.savez("./Argu_Result_npz/acc-1.npz", test_acc_array, train_acc_array)
        np.savez("./Argu_Result_npz/loss-1.npz", test_loss_array, train_loss_array)
        np.savez("./Argu_Result_npz/pred-1_+"+numberOfnet+".npz",prediction_array)

    # plt_x = np.arange(0, num_steps+microtrain_steps, 100)
    # plt_y = train_loss_array
    #
    # plt.figure(figsize=(8, 4))
    # plt.plot(plt_x, plt_y, label="train_loss", color="red", linewidth=2)
    # plt_y = test_loss_array
    # plt.plot(plt_x, plt_y, label="test_loss", color="blue", linewidth=2)
    # plt.legend(loc='upper right')
    # plt.xlabel("iterations",size=20)
    # plt.ylabel("loss",size=20)
    # s = "num of steps: "+np.str(num_steps) +"  learning rate:  "+np.str(learning_rate)+"  batch size:  "+np.str(batch_size)+"  microtrain_steps: "+np.str(microtrain_steps)
    # plt.title(s,size=15)
    #
    # plt.figure(figsize=(8, 4))
    # plt_y = train_acc_array
    # plt.plot(plt_x, plt_y, label="train_acc", color="red", linewidth=2)
    # plt_y = test_acc_array
    # plt.plot(plt_x, plt_y, label="test_acc", color="blue", linewidth=2)
    # plt.legend(loc='upper left')
    # plt.xlabel("iterations",size=20)
    # plt.ylabel("acc",size=20)
    # plt.title(s, size=15)
    # plt.show()
