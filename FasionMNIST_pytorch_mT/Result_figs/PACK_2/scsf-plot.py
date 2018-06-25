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

k_list = [210, 220, 240, 260, 280, 290]
c_list = [16, 32, 48, 64, 80, 96, 112, 128]


for k in k_list[5:6]:
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for c in c_list:
        file_name = 'Acc_'+str(k)+'.npz'
        full_name = result_dir + '/C' + str(c) + 'F10/' + file_name
        try:
            test_acc = load_test_acc(full_name)
        except:
            continue
        axis_x = [c]*300
        ax.plot(axis_x, epochs, test_acc, label='C='+str(c))  # 绘制数据点
    
    ax.set_title('K='+str(k))
    ax.set_zlabel('test_acc')  # 坐标轴
    ax.set_ylabel('epoch')
    ax.set_xlabel('C')
    plt.legend(loc='best')
    plt.show()
    #plt.savefig(img_dir + '/' + dir_name + '.jpg',dpi=1000)

