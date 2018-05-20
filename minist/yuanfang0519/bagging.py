from collections import Counter
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def read_file(filename):
    temp = np.load(filename)
    return temp["arr_0"]

def analysis_result():
    end_pred = np.zeros((5, 10000))

    for i in range(1,6):
        pred = read_file("./yuanfang0518/Argu_Result_npz/micro_pred20_"+str(i)+".npz")
        pred = np.reshape(pred,[10000])
        acc =0
        acc = np.mean(np.equal(pred, np.argmax(mnist.test.labels, 1)))
        print("net"+str(i)+" result:",acc)
        end_pred[i-1] = pred
    temp = np.zeros(10000)
    for j in range(10000):
        word_counts = Counter(end_pred[:, j])
        temp[j] = word_counts.most_common(1)[0][0]
    acc = np.mean(np.equal(temp, np.argmax(mnist.test.labels, 1)))
    print("bagging_result:{%.5f}"%acc)
analysis_result()