# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import glob
import re
import os

from matplotlib import cm


def load_test_acc(filename):
    temp = np.load(filename)
    return temp["arr_0"]

result_dir = 'Result_npz'
img_dir = 'Result_figs2'
dir_list = os.listdir(result_dir)
# dir_list.sort(key=lambda x:int(x[1:].split('F')[0]))

steps = [0, 1, 5] + list(range(10,301,10))
epochs = list(range(1,301))
c_list = [16, 32, 48, 64, 80, 96, 112, 128]
# c_list2 = np.arange(192,1025,64)
c_list2=list(range(192,1025,64))
c_list +=c_list2

XX,YY=np.meshgrid(steps, epochs)

for c in c_list:
    ZZ = []
    # fig = plt.figure(figsize=(15,5))
    dir_name = 'SCSF_C'+str(c)+'F10'
    # ax = fig.gca(projection='3d')
    max_step = -1
    max_epoch = -1
    max_acc = -1
    for step in steps:
        file_name = 'Acc_'+str(step)+'.npz'
        full_name = result_dir + '/' + dir_name + '/' + file_name
        test_acc = load_test_acc(full_name)
        if np.max(test_acc) > max_acc:
           max_acc = np.max(test_acc)
           max_epoch = epochs[np.argmax(test_acc)]
           max_step = step
        ZZ.append(test_acc)
    print('when kernel_num: '+str(c)+'\t max accuracy is '+str(max_acc)+' \t K step: '+str(max_step)+' \t epoch: '+str(max_epoch))
    '''
    ZZ=np.array(ZZ).T
    surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('K_microTrain')
    ax.set_ylabel('Epochs')
    ax.set_zlabel('Test Accuracy')
    ax.set_title('Kernel Num = '+ str(c))
    dir_name="Kernel_"+str(c)
    plt.savefig(img_dir + '/' + dir_name + '.png',dpi=1000)
    #plt.savefig(img_dir + '/' + dir_name + '.jpg',dpi=1000)
    '''
