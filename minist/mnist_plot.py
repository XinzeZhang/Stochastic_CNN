import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.interpolate import spline

kernel_num = str(64) #卷积核个数
is_acc=0    #为1时为loss，为0时为acc
is_loss =1  #为1时为loss，为0时为acc

def load_data(filename):
    temp = np.load(filename)
    return temp["arr_0"],temp["arr_1"]

def acc_micro(s,loc,micro_epochs,color):
    test1,train1 = load_data("./Result_npz/"+kernel_num+"/"+s+str(micro_epochs)+".npz")
    size = np.size(test1)
    plt_x = np.arange(1, 100 + 1, 1)
    plt_y = test1[:100]
    max_index=np.argmax(plt_y) # find the max point of test sets
    plt.plot(plt_x, plt_y, label="Test "+s, color=color, linewidth=1,linestyle="--")
    plt.plot(max_index,plt_y[max_index], color=color, marker='s')
    # show_max='['+str(max_index)+' '+str(plt_y[max_index])+']'
    show_max='['+str(max_index)+',%.5f]' %(plt_y[max_index])
    plt.annotate(show_max,xytext=(max_index,plt_y[max_index]),xy=(max_index,plt_y[max_index]),fontsize=10)
    
    plt_y = train1[:100]
    plt.plot(plt_x, plt_y, label="Train "+s, color=color, linewidth=1,linestyle="-")

def loss_micro(s,loc,micro_epochs,color):
    test1,train1 = load_data("./Result_npz/"+kernel_num+"/"+s+str(micro_epochs)+".npz")
    size = np.size(test1)
    plt_x = np.arange(1, 100 + 1, 1)
    plt_y = test1[:100]
    min_index=np.argmin(plt_y) # find the max point of test sets
    plt.plot(plt_x, plt_y, label="Test "+s, color=color, linewidth=1,linestyle="--")
    plt.plot(min_index,plt_y[min_index], color=color, marker='s')
    show_max='['+str(min_index)+',%.5f]' %(plt_y[min_index])
    plt.annotate(show_max,xytext=(min_index,plt_y[min_index]),xy=(min_index,plt_y[min_index]),fontsize=10)
    
    plt_y = train1[:100]
    plt.plot(plt_x, plt_y, label="Train "+s, color=color, linewidth=1,linestyle="-")


def plot_micro(type=0):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Epoch", size=14)
    k_microTrain=10
    if type ==0 :
        s = "Acc"
        loc = "lower right"
        acc_micro(s,loc,k_microTrain,"red")
        plt.legend(loc=loc)
        # new_micro(s,loc,20,"blue")
        plt.ylabel("Acc", size=14)

    else:
        s ="Loss"
        loc = "upper right"
        loss_micro(s,loc,k_microTrain,"red")
        plt.legend(loc=loc)
        # new_micro(s,loc,20,"blue")
        plt.ylabel("Loss", size=14)
    
    plt.savefig('./Result_fig/SCSF_'+str(k_microTrain)+'mT_'+s+'.png')

if __name__ == '__main__':
    plot_micro(is_loss)
    # plt.show()
    plot_micro(is_acc)
    # plt.show()