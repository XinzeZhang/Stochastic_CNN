import matplotlib.pyplot as plt
# import matplotlib as mpl
import numpy as np
import sys
from scipy.interpolate import spline



def load_data(filename):
    temp = np.load(filename)
    return temp["arr_0"],temp["arr_1"]



if __name__ == '__main__':

    for mt in (0,1,5,10,15,20,25,30,35,40):
        test1,train1 =load_data("./C32_C64_F1024_F10/Argu_Result_npz_lastTwo/acc"+str(mt)+"_1.npz")
        # print(test1[0])

        test2,train2 =load_data("./C32_C64_F1024_F10/Argu_Result_npz_lastTwo/acc"+str(mt)+"_2.npz")
        # print(test2[0])

        test3,train3 =load_data("./C32_C64_F1024_F10/Argu_Result_npz_lastTwo/acc"+str(mt)+"_3.npz")
        # print(test3[0])

        test4,train4 =load_data("./C32_C64_F1024_F10/Argu_Result_npz_lastTwo/acc"+str(mt)+"_4.npz")
        # print(test4[0])

        test5,train5 =load_data("./C32_C64_F1024_F10/Argu_Result_npz_lastTwo/acc"+str(mt)+"_5.npz")
        # print(test5[0])


        test=test1+test2+test3+test4+test5
        train=train1+train2+train3+train4+train5

        test_average=test/5
        train_average=train/5
        print(mt, test_average[-1].round(4))

        # np.savez("./C32_C64_F1024_F10/Argu_Result_npz/acc_"+str(mt)+"_average.npz",test_average,train_average)