from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np

from cnn_model import SCSF_Net, DCDF_Net
from train import train,micro_train


# Training settings
# ==============================================================================
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--total_epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--k_allTrain_epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--microTrain_epochs', type=int, default=298, metavar='N',
                    help='number of epochs to train (default: 200)')
# ------------------------------------------------------------------------------
parser.add_argument('--SGD_lr', type=float, default=0.001, metavar='LR',
                    help='SGD learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.995, metavar='M',
                    help='SGD momentum (default: 0.995)')
parser.add_argument('--aT_Adam_lr', type=float, default=0.0001, metavar='LR',
                    help='Adam learning rate (default: 0.001)')
parser.add_argument('--mT_Adam_lr', type=float, default=0.0001, metavar='LR',
                    help='Adam learning rate (default: 0.0001)')                    
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
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device=torch.device("cuda" if use_cuda else "cpu")

    aT_model = SCSF_Net()
    if os.path.exists("./Model_State/"+str(aT_model.__class__.__name__)) == False:
        os.mkdir("./Model_State/"+str(aT_model.__class__.__name__))
    # model.share_memory().to(device) # gradients are allocated lazily, so they are not shared here 


    train_acc_array,test_acc_array=train(args,aT_model,device)

    torch.save(aT_model.state_dict(),"./Model_State/"+str(aT_model.__class__.__name__)+"_"+str(args.k_allTrain_epochs)+".pkl")

    mT_model=SCSF_Net()
    mT_model.load_state_dict(torch.load("./Model_State/"+str(aT_model.__class__.__name__)+"_"+str(args.k_allTrain_epochs)+".pkl"))

    layer_id=0
    for child in mT_model.children():
        layer_id +=1
        print("layer Id: "+str(layer_id), child)
        for param in child.parameters():
            param.requires_grad = False

    mT_model.fc1.weight.requires_grad= True
    mT_model.fc1.bias.requires_grad=True

    # for param in model_re.parameters():
    #     print(param)
    #     param.requires_grad = False
    
    # num_ftrs=model_re.fc1.in_features
    # model_re.fc1=nn.Linear(num_ftrs,10)

    mT_train_acc_array,mT_test_acc_arrary = micro_train(args,mT_model,device)

    if os.path.exists("./Result_npz/lastOne") == False:
        os.mkdir("./Result_npz/lastOne")
    dirs = "./Result_npz/lastOne"
    if not os.path.exists(dirs):
        os.mkdir(dirs)
    # np.savez(dirs+"/acc"+str(int(microtrain_steps/display_step))+".npz", test_acc_array, train_acc_array)
    np.savez(dirs+"/Acc_"+str(args.k_allTrain_epochs)+".npz", test_acc_array, train_acc_array)