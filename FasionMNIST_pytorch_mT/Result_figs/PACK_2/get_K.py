# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import glob
import re
import os

def load_test_acc(filename):
    temp = np.load(filename)
    return temp["arr_0"]

result_dir = 'Result_npz'
img_dir = 'imgs'
dir_list = os.listdir(result_dir)
dir_list.sort(key=lambda x:(x[1:].split('F')[0]))

steps = [0, 1, 5] + list(range(10,301,10))
epochs = list(range(1,301))

for dir_name in dir_list[:]:
    score_list = []
    for step in steps:
        file_name = 'Acc_'+str(step)+'.npz'
        full_name = result_dir + '/' + dir_name + '/' + file_name
        try:
            test_acc = load_test_acc(full_name)
        except:
            break
        axis_x = [step]*300
        score_list.append(test_acc[-1])
    max_index = np.argmax(np.array(score_list))
    # if max_index != len(score_list) - 1:
    print (dir_name +'\t K=' + str(steps[max_index]) + ' acc: ' + str(score_list[max_index])+'\t K=' + str(steps[-1]) + ' acc: ' + str(score_list[-1]))
