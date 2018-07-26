# coding: utf-8
import matplotlib
matplotlib.use('Agg')
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
# img_dir = 'Result_figs1'
dir_list = os.listdir(result_dir)
# dir_list.sort(key=lambda x:int(x[1:].split('F')[0]))

steps = [0, 1, 5] + list(range(10,301,10))
epochs = list(range(1,301))

c_list = [16, 32, 48, 64, 80, 96, 112, 128]
# c_list2 = np.arange(192,1025,64)
c_list2=list(range(192,1025,64))
c_list +=c_list2

XX,YY=np.meshgrid(c_list,epochs)

for k in steps:
    max_kernel = -1
    max_epoch = -1 
    max_acc = -1
    ZZ = []
    # fig = plt.figure(figsize=(15,5))
    # ax = fig.gca(projection='3d')  # 创建一个三维的绘图工程
    for c in c_list:
        file_name = 'Acc_'+str(k)+'.npz'
        full_name = result_dir + '/SCSF_C' + str(c) + 'F10/' + file_name
        try:
            test_acc = load_test_acc(full_name)
            if np.max(test_acc) > max_acc:
                max_acc = np.max(test_acc)
                max_epoch = epochs[np.argmax(test_acc)]
                max_kernel = c
        except:
            continue
        ZZ.append(test_acc)
    print('when K step: '+str(k)+' \t max accuracy is '+str(max_acc)+' \t Kernel num: '+str(max_kernel)+' \t epoch: '+str(max_epoch))
