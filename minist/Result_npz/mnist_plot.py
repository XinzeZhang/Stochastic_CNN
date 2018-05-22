import matplotlib.pyplot as plt
# import matplotlib as mpl
import numpy as np
import sys
from scipy.interpolate import spline



# kernel_num = str(64) #卷积核个数
kernel_num = "C32_C64_F1024_F10" #卷积核个数
is_acc=0    #为1时为loss，为0时为acc
is_loss =1  #为1时为loss，为0时为acc

def load_data(filename):
    temp = np.load(filename)
    return temp["arr_0"],temp["arr_1"]

def plot_micro(type=0):
    plt.figure(figsize=(15, 8))
    plt.xlabel("Epoch", size=14)
    # K_microTrain=[0,10,20,25,35,40]
    # mpl.style.use('seaborn-bright')
    # cmap=plt.cm.Spectral
    # Color_microTrain=['yellow','red','blue','cyan','gray','black']
    K_color={"0":"C1","1":"C2","5":"C3","10":"C4","15":"C5","20":"C6","25":"C7","30":"C8","40":"C9"}
    # K_color={"0":cmap(0.0),"10":cmap(0.2),"20":cmap(0.4),"25":cmap(0.6),"35":cmap(0.8),"40":cmap(1.)}
    # K_color={"0":"xkcd:bronze","10":"xkcd:pinky red","20":"xkcd:dull red","25":"xkcd:marine","35":"xkcd:bluish","40":"xkcd:pale teal"}
    # k_microTrain=10
    best_acc=''
    if type ==0 :
        indicator = "acc"
        loc = "lower right"
        for key,value in K_color.items():
            best_acc+=acc_micro(indicator,loc,key,value)
        # acc_micro(indicator,loc,k_microTrain,"red")
        plt.legend(loc=loc)
        # new_micro(indicator,loc,20,"blue")
        plt.ylabel("Acc", size=14)
        best_acc=best_acc.rstrip()
        plt.annotate(best_acc, xy=(1, 0.66), xytext=(-10, -0), fontsize=10,
    xycoords='axes fraction', textcoords='offset points',
    bbox=dict(facecolor='white', alpha=0.2),
    horizontalalignment='right', verticalalignment='top')

    # else:
    #     indicator ="Loss"
    #     loc = "upper right"
    #     loss_micro(indicator,loc,k_microTrain,"red")
    #     plt.legend(loc=loc)
    #     # new_micro(indicator,loc,20,"blue")
    #     plt.ylabel("Loss", size=14)
    
    # plt.savefig('./Result_fig/SCSF_'+str(k_microTrain)+'mT_'+indicator+'.png')
    # plt.savefig('./Result_fig/DCDF_'+indicator+'.png')

def acc_micro(indicator,loc,micro_epochs,color):
    # test1,train1 = load_data("./Result_npz/"+kernel_num+"/"+indicator+str(micro_epochs)+".npz")
    test1,train1 = load_data("./C32_C64_F1024_F10/"+indicator+str(micro_epochs)+".npz")
    size = np.size(test1)
    plt_x = np.arange(1, 100 + 1, 1)
    test_y = test1[:100]
    max_index=np.argmax(test_y) # find the max point of test sets

    if micro_epochs=="0":
        label = "allTrain"
    elif micro_epochs=="-1":
        label ="lastLayer"
    else:
        label = str(micro_epochs)+"mT"

    plt.plot(plt_x, test_y, label=label+" Test "+indicator, color=color, linewidth=1,linestyle="--")
    plt.plot(max_index,test_y[max_index], color=color, marker='o')
    # show_max='['+str(max_index)+' '+str(plt_y[max_index])+']'
    show_max='['+str(max_index)+',%.3f]' %(100.0 * test_y[max_index])
    plt.annotate(show_max,xytext=(max_index,test_y[max_index]),xy=(max_index,test_y[max_index]),fontsize=10, color=color)
    
    # train_y = train1[:100]
    # plt.plot(plt_x, train_y, label=label+" Train "+indicator, color=color, linewidth=1,linestyle="-")
    return label+" Test "+indicator+': '+str(max_index)+', %.3f\n' %(100.0 * test_y[max_index])

def loss_micro(indicator,loc,micro_epochs,color):
    test1,train1 = load_data("./C32_C64_F1024_F10/"+indicator+str(micro_epochs)+".npz")
    size = np.size(test1)
    plt_x = np.arange(1, 100 + 1, 1)
    plt_y = test1[:100]
    min_index=np.argmin(plt_y) # find the max point of test sets
    
    if micro_epochs=='0':
        label = "allTrain"
    elif micro_epochs=='-1':
        label ="lastLayer"
    else:
        label = str(micro_epochs)+"mT"

    plt.plot(plt_x, plt_y, label=label+" Test "+indicator, color=color, linewidth=1,linestyle="--")
    plt.plot(min_index,plt_y[min_index], color=color, marker='o')
    show_max='['+str(min_index)+',%.5f]' %(plt_y[min_index])
    plt.annotate(show_max,xytext=(min_index,plt_y[min_index]),xy=(min_index,plt_y[min_index]),fontsize=10, color=color)
    
    plt_y = train1[:100]
    plt.plot(plt_x, plt_y, label=label+" Train "+indicator, color=color, linewidth=1,linestyle="-")

if __name__ == '__main__':
    # plot_micro(is_loss)
    # plt.show()
    plot_micro(is_acc)
    plt.show()