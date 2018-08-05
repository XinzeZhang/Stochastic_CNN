# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import glob
import re
import os

def load_test_acc(filename):
    temp = np.load(filename)
    return temp["arr_0"]

result_dir = 'Result_npz'
img_dir = 'Result_figs/PACK_2'
dir_list = os.listdir(result_dir)
# dir_list.sort(key=lambda x:int(x[1:].split('F')[0]))

# steps = [0, 1, 5] + list(range(10,301,10))
# steps = list(range(230,301,10))
steps = [220]
epochs = list(range(1,301))

k_list = [210, 220, 240, 260, 280, 290]
c_list = [16, 32, 48, 64, 80, 96, 112, 128]
# c_list2 = np.arange(192,1025,64)
c_list2=list(range(192,1025,64))
c_list +=c_list2

XX,YY=np.meshgrid(c_list,epochs)


for k in steps:
    ZZ = []
    fig = plt.figure(figsize=(15,5))
    ax = fig.gca(projection='3d')  # 创建一个三维的绘图工程
    for c in c_list:
        file_name = 'Acc_'+str(k)+'.npz'
        full_name = result_dir + '/SCSF_C' + str(c) + 'F10/' + file_name
        try:
            test_acc = load_test_acc(full_name)
        except:
            print(full_name + " not found!")
            continue
        ZZ.append(test_acc)
    ZZ=np.array(ZZ).T
    surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('Convolution kernels')
    ax.set_ylabel('Epochs')
    ax.set_zlabel('Test Accuracy')
    ax.set_title('K_microTrain when k = '+ str(k))
    dir_name="K_"+str(k)
    plt.savefig(img_dir + '/' + dir_name + '.png',dpi=1000)

    # plt.show()
    

'''
for k in k_list[5:6]:
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for c in c_list:
        file_name = 'Acc_'+str(k)+'.npz'
        full_name = result_dir + '/SCSF_C' + str(c) + 'F10/' + file_name
        try:
            test_acc = load_test_acc(full_name)
        except:
            continue
        axis_x = [c]*300
        ax.plot(axis_x, epochs, test_acc, label='C='+str(c))  # 绘制数据点
    
    ax.set_title('K='+str(k))
    ax.set_zlabel('Test Accuracy')  # 坐标轴
    ax.set_ylabel('Epochs')
    ax.set_xlabel('Convolution kernels')
    # plt.legend(loc='best')
    plt.show()
    #plt.savefig(img_dir + '/' + dir_name + '.jpg',dpi=1000)
'''