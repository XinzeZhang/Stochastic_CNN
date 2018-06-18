from _data_process import asMinutesUnit
import numpy  as np

f = np.loadtxt('D:/xinze/Documents/GitHub/Stochastic_CNN/FasionMNIST_pytorch_mT/Result_npz/C32F10/Time_Log.txt')

f[:,2]=f[:,1]*60+f[:,2]
for k , t in zip(f[:,0],f[:,2]):
    m=asMinutesUnit(t)
    print("%d\t%s" % (k,m) )

with open("./Time_Log.txt", "w+") as n:
    for k , t in zip(f[:,0],f[:,2]):
        m=asMinutesUnit(t)
        print("%d\t%s" % (k,m) , file=n)
