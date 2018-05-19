from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
import math
import sys
import os
import seaborn as sns
# Import MNIST data
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST-data/", one_hot=True)
os.environ['CUDA_VISIBLE_DEVICES']='0'
if os.path.exists("./Result_npz_BN") == False:
    os.mkdir("./Result_npz_BN")
learning_rate = 0.0001

batch_size = 64
display_step = int(math.ceil(55000/batch_size))
microtrain_steps = display_step*int(sys.argv[1])
num_steps = display_step*(200-int(sys.argv[1]))


# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


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
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

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
    fc1= tf.nn.batch_normalization(fc1,0.001,1.0,0,1,0.0001)
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
        'wc1': tf.Variable(np.array(np.random.normal(0,0.1,[5,5,1,numC1]),np.float32)),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(np.array(np.random.normal(0,0.1,[5,5,numC1,numC2]),np.float32)),
        # # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(np.array(np.random.normal(0,0.1,[7*7*numC2,num_fc1]),np.float32)),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(np.array(np.random.normal(0,0.1,[num_fc1,num_classes]),np.float32))
    }

    biases = {
        'bc1': tf.Variable(tf.constant(0.1, shape=[numC1]) ,dtype=np.float32),
        'bc2': tf.Variable(tf.constant(0.1, shape=[numC2]), dtype=np.float32),
        'bd1': tf.Variable(tf.constant(0.1, shape=[num_fc1]), dtype=np.float32),
        'out': tf.Variable(tf.constant(0.1, shape=[num_classes]), dtype=np.float32)
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
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    start = time.time()
    e_loss=0
    e_acc =0

    for step in range(1, microtrain_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        _,t_loss,t_acc = sess.run([train_op,loss_op,accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})

        if step % display_step == 0 :
            # Calculate batch loss and accuracy
            train_loss = 0
            train_acc = 0

            for i in range(100):  # calculate the train acc and loss
                start_index = i * 550
                end_index = (i + 1) * 550
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: mnist.train.images[start_index:end_index],
                                                                     Y: mnist.train.labels[start_index:end_index],
                                                                     keep_prob: 1.0})
                # print(loss,acc)
                train_loss += loss
                train_acc += acc

            train_loss_array.append(train_loss / 100)
            train_acc_array.append(train_acc / 100)

            print("Epoch " + str(int(step/display_step)) + ", Minibatch Loss= " + \
                  "{:.4f}".format(train_loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(train_acc))

            acc =0
            loss=0
            test_acc=0
            test_loss=0
            for i in range(10):# calculate the test acc and loss
                start_index = i * 1000
                end_index = (i + 1) * 1000
                acc,loss= sess.run([accuracy,loss_op], feed_dict={X: mnist.test.images[start_index:end_index],
                                                     Y: mnist.test.labels[start_index:end_index],
                                                     keep_prob: 1.0})
                test_loss+=loss
                test_acc+=acc
            test_loss_array.append(test_loss / 10)
            test_acc_array.append(test_acc / 10)
            print("Epoch " + str(int(step/display_step)) + ", Minibatch Loss= " + \
                  "{:.4f}".format(test_loss/10) + ", test Accuracy= " + \
                  "{:.5f}".format(test_acc/10))
    print("cost_time: %.6f"% (time.time() - start))
    with open("./Result_npz_BN/outputlog.txt", "a+") as f:
        print("----------------------------------------------",file=f)
        print("cost_time: %.6f" % (time.time() - start),file=f)

    print("Optimization Finished!")

    acc = 0
    for i in range(10):
        start_index = i * 1000
        end_index = (i + 1) * 1000
        acc += sess.run(accuracy, feed_dict={X: mnist.test.images[start_index:end_index],
                                             Y: mnist.test.labels[start_index:end_index],
                                             keep_prob: 1.0})

    print('%.6f' % (acc / 10))
    temp_weights, temp_biases = sess.run([weights, biases])



#start the lastlayer train

weights2 = {
    'wc1': tf.Variable(temp_weights['wc1'],trainable=False),
    'wc2': tf.Variable(temp_weights['wc2'],trainable=False),
    'wd1': tf.Variable(temp_weights['wd1'],trainable=False),
    'out': tf.Variable(temp_weights['out'])
}

biases2 = {
    'bc1': tf.Variable(temp_biases['bc1'],trainable=False),
    'bc2': tf.Variable(temp_biases['bc2'],trainable=False),
    'bd1': tf.Variable(temp_biases['bd1'],trainable=False),
    'out': tf.Variable(temp_biases['out'])
}
logits2 = conv_net(X, weights2, biases2, keep_prob)

pred2 = tf.nn.softmax(logits2)
# Define loss and optimizer
cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=Y))
optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost2)

# Evaluate model
correct_pred2 = tf.equal(tf.argmax(pred2,1),tf.arg_max(Y,1))
accuracy2 = tf.reduce_mean(tf.cast(correct_pred2, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    start = time.time()
    e_loss=0
    e_acc =0
    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        _,t_loss,t_acc = sess.run([optimizer2,cost2,accuracy2], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})

        if step % display_step == 0 :
            # Calculate batch loss and accuracy
            train_loss = 0
            train_acc = 0

            for i in range(100):  # calculate the train acc and loss
                start_index = i * 550
                end_index = (i + 1) * 550
                loss, acc = sess.run([cost2, accuracy2], feed_dict={X: mnist.train.images[start_index:end_index],
                                                                     Y: mnist.train.labels[start_index:end_index],
                                                                     keep_prob: 1.0})
                # print(loss,acc)
                train_loss += loss
                train_acc += acc

            train_loss_array.append(train_loss / 100)
            train_acc_array.append(train_acc / 100)

            test_acc = 0
            test_loss = 0
            for i in range(10):
                start_index = i * 1000
                end_index = (i + 1) * 1000
                acc, loss = sess.run([accuracy2, cost2], feed_dict={X: mnist.test.images[start_index:end_index],
                                                                     Y: mnist.test.labels[start_index:end_index],
                                                                     keep_prob: 1.0})
                test_loss += loss
                test_acc += acc
            test_loss_array.append(test_loss / 10)
            test_acc_array.append(test_acc / 10)
            if(step % display_step ==0) :
                print("Epoch " + str(int(step/display_step)) + ", Minibatch Loss= " + \
                  "{:.6f}".format(test_loss / 10) + ", test Accuracy= " + \
                  "{:.6f}".format(test_acc / 10) +"  time= "+"{:.6f}".format(time.time()-start))
    print("cost_time: %.6f" % (time.time() - start))
    with open("./Result_npz_BN/outputlog.txt", "a+") as f:
        print("cost_time: %.6f" % (time.time() - start), file=f)
    print("Optimization Finished!")
    acc = 0
    for i in range(10):
        start_index = i * 1000
        end_index = (i + 1) * 1000
        acc += sess.run(accuracy2, feed_dict={X: mnist.test.images[start_index:end_index],
                                             Y: mnist.test.labels[start_index:end_index],
                                             keep_prob: 1.0})

    print('%.6f'%(acc / 10))


    np.savez("./Result_npz_BN/acc"+str(int(microtrain_steps/display_step))+".npz", test_acc_array, train_acc_array)
    np.savez("./Result_npz_BN/loss"+str(int(microtrain_steps/display_step))+".npz", test_loss_array, train_loss_array)
