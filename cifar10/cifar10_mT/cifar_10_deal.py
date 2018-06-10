import math
import cifar10
import cifar10_input
import tensorflow as tf
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

kernel_num = int(sys.argv[1])
is_var = int(sys.argv[2])
lam = 0.01
print(lam)
os.environ['CUDA_VISIBLE_DEVICES']='0'
train_size = 50000

# GD_epochs = 100 
epochs = 200
batch_size =128
iterations_of_epoch = int(math.ceil(train_size/batch_size))

data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

cifar10.maybe_download_and_extract()

images_train,labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
images_train_not_epoch,labels_train_not_epoch = cifar10_input.inputs(eval_data=False,data_dir=data_dir,batch_size=batch_size)
images_test ,labels_test = cifar10_input.inputs(eval_data=True, data_dir= data_dir,batch_size=batch_size)

image_holder = tf.placeholder(tf.float32,[batch_size,24,24,3])
label_holder = tf.placeholder(tf.int32,[batch_size])
learning_rate = tf.placeholder(tf.float32)

def _variable_on_cpu(stddev, shape,trainable =1):
    dtype =tf.float32
    is_train = True if trainable else False
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev,dtype=dtype),trainable=is_train)
    return var

def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return x
def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k+1,k+1,1],strides=[1,k,k,1],padding='SAME')
def conv_net(image_holder,weights,biases):
    conv1 = tf.nn.relu(conv2d(image_holder,weights['wc1'],biases['bc1']))
    pool1 = maxpool2d(conv1)
    #
    # conv2 = tf.nn.relu(conv2d(pool1,weights['wc2'],biases['bc2']))
    # pool2 = maxpool2d(conv2)

    fc1 = tf.reshape(pool1, [-1, weights['out'].get_shape().as_list()[0]])
    # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # fc1 = tf.nn.relu(fc1)

    # fc1 = tf.nn.dropout(fc1, dropout)

    # fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    # fc2 = tf.nn.relu(fc2)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def loss(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'),name='total_loss')


weights = {
    'wc1': _variable_on_cpu(0.01,[5,5,3,kernel_num],trainable=is_var),
    # 'wc2': constant_variable_weight([5,5,32,64],stddev=5e-2),
    # 'wd1': constant_variable_weight([12*12*32,1024],stddev=0.04),
    # # 'wd2': constant_with_weight_loss([384,192],stddev=0.04),
    # 'out': variable_with_weight_loss(shape = [12*12*kernel_num, 10], stddev = 1/192.0, wl = 0.004)
    'out': _variable_on_cpu(1/192.0,[12*12*kernel_num,10]),
}

biases = {
    'bc1': _variable_on_cpu(0.01,[kernel_num],trainable=is_var),
    # 'bc2': constant_variable_biases(0.1,shape=[64]),
    # 'bd1': constant_variable_biases(0.1,shape=[1024]),
    # # 'bd2': constant_variable_biases(0.1,shape=[192]),
    'out': _variable_on_cpu(0.1,[10])
}
logits = conv_net(image_holder,weights,biases)
loss = loss(logits,label_holder)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
top_k_op = tf.nn.in_top_k(logits,label_holder,1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()


with open("./Result_1L/outputlog.txt", "a+") as f:
    print("kernel_num=%d   is_var = %d---------------------------------------------\n"%(kernel_num,is_var),file=f)
#训练开始

train_time = time.time()
start_time = time.time()
global train_loss,train_acc,test_loss,test_acc
test_loss_array = []
train_loss_array = []
test_acc_array = []
train_acc_array = []
cnt =0
lr = 0.001
for i_epoch in range(epochs):
    true_count =0
    loss_count =0
    total_sample_count = iterations_of_epoch*batch_size
    if i_epoch == 100 or i_epoch==200:
        lr= lr/10
    for iterations in range(iterations_of_epoch):
        cnt+=1
        image_batch,label_batch = sess.run([images_train,labels_train])
        _, train_loss,train_acc = sess.run([train_op, loss,top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch,learning_rate:lr})

    for i in range(iterations_of_epoch):
        image_batch,label_batch = sess.run([images_train_not_epoch,labels_train_not_epoch])
        train_loss, train_acc = sess.run([ loss, top_k_op],
                                         feed_dict={image_holder: image_batch, label_holder: label_batch})
        true_count += np.sum(train_acc)
        loss_count += train_loss
    duration = time.time() - start_time
    train_acc = true_count/ total_sample_count
    train_loss = loss_count /iterations_of_epoch
    train_loss_array.append(train_loss)
    train_acc_array.append(train_acc)
    # 测试开始
    num_examples = 10000
    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    loss_count = 0
    total_sample_count = num_iter * batch_size
    step = 0
    while step < num_iter:
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions ,test_loss= sess.run([top_k_op,loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
        true_count += np.sum(predictions)
        loss_count += test_loss
        step += 1
    test_acc = true_count / total_sample_count
    test_loss = loss_count / num_iter
    test_loss_array.append(test_loss)
    test_acc_array.append(test_acc)
    print('epoch =%d test_acc =%.5f' % (i_epoch,test_acc))
    with open("./Result_1L/outputlog.txt", "a+") as f:
    #    print('epoch =%d test_loss =%.5f train_loss = %.5f' % (i_epoch, test_loss, train_loss), file=f)
        print('epoch =%d test_acc =%.5f train_acc = %.5f' % (i_epoch, test_acc,train_acc),file=f)

print('cnt = ',cnt)
s =''
if is_var:
    s = s+'allTrained'
else:
    s = s+'lastLayer'
print('train_time: ',time.time() - train_time)
np.savez("./Result_1L/"+s+"/acc"+str(kernel_num)+".npz",test_acc_array,train_acc_array)
np.savez("./Result_1L/"+s+"/loss"+str(kernel_num)+".npz",test_loss_array,train_loss_array)

