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
dir_list.sort(key=lambda x:int(x[1:].split('F')[0]))

steps = [0, 1, 5] + list(range(10,301,10))
epochs = list(range(1,301))

for dir_name in dir_list[7:8]:
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for step in steps:
        file_name = 'Acc_'+str(step)+'.npz'
        full_name = result_dir + '/' + dir_name + '/' + file_name
        test_acc = load_test_acc(full_name)
        axis_x = [step]*300
        ax.plot(axis_x, epochs, test_acc, label=dir_name+'_K='+str(step))  # 绘制数据点
    
    ax.set_title(dir_name)
    ax.set_zlabel('test_acc')  # 坐标轴
    ax.set_ylabel('epoch')
    ax.set_xlabel('K_mt')
    plt.legend(loc='best')
    plt.show()
    #plt.savefig(img_dir + '/' + dir_name + '.jpg',dpi=1000)

