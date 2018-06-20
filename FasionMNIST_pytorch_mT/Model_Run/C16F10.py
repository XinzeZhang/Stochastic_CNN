from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np

from _cnn_model import C16F10,weights_init
from _mixed_train_sgd import train,micro_train,all_train

import time
from _data_process import asMinutesUnit

# Training settings
# ==============================================================================
parser = argparse.ArgumentParser(description='PyTorch FasionMNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--total_epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--k_allTrain_epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 100)')
# ------------------------------------------------------------------------------
# setting learning rate as https://cs.nyu.edu/~wanli/dropc/dropc.pdf.
parser.add_argument('--LR_window1', type=int, default=100, metavar='W1',
                    help='Adam learning rate window 1 (default: 150)')
parser.add_argument('--LR_window2', type=int, default=50, metavar='W2',
                    help='Adam learning rate window 2 (default: 100)')
parser.add_argument('--LR_window3', type=int, default=25, metavar='W3',
                    help='Adam learning rate window 3 (default: 50)')

parser.add_argument('--SGD_lr', type=float, default=0.001, metavar='LR',
                    help='SGD learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--Adam_lr', type=float, default=0.001, metavar='LR',
                    help='Initial Adam learning rate (default: 0.01)')

# ------------------------------------------------------------------------------
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# ------------------------------------------------------------------------------
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=60, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_train', type=int, default=1, metavar='N',
                    help='how many training times to use (default: 1)')
# ==============================================================================

if __name__ == '__main__':


    args = parser.parse_args()

    torch.manual_seed(args.seed)

    args.total_epochs = args.LR_window1 + args.LR_window2 *2 + args.LR_window3*4

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device=torch.device("cuda" if use_cuda else "cpu")

    aT_model = C16F10()
    aT_model.apply(weights_init)
    # aT_model.share_memory().to(device) # gradients are allocated lazily, so they are not shared here 
    if os.path.exists("../Model_State/"+str(aT_model.__class__.__name__)) == False:
        os.mkdir("../Model_State/"+str(aT_model.__class__.__name__))
    if os.path.exists("../Result_npz/"+str(aT_model.__class__.__name__)) == False:
        os.mkdir("../Result_npz/"+str(aT_model.__class__.__name__))
        
    time_start=time.time()
    if args.k_allTrain_epochs == args.total_epochs:
        train_acc_array,test_acc_array=all_train(args,aT_model,device)
    elif 0<=args.k_allTrain_epochs < args.total_epochs:
        train_acc_array,test_acc_array=micro_train(args,aT_model,device)
    else :
        print("Error ! please make sure k_allTrain show be smaller than total!")
        exit()


    log_dirs = "../Result_npz/"+str(aT_model.__class__.__name__)
    
    with open(log_dirs+"/Time_Log.txt", "a+") as f:
        print("%d\t%s" % (args.k_allTrain_epochs,asMinutesUnit(time.time() - time_start)) , file=f)
    # np.savez(dirs+"/acc"+str(int(microtrain_steps/display_step))+".npz", test_acc_array, train_acc_array)
    np.savez(log_dirs+"/Acc_"+str(args.k_allTrain_epochs)+".npz", test_acc_array, train_acc_array)