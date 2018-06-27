import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np


def train(args, model, device):
    model = model.to(device)
    model.share_memory().to(device)
    if os.path.exists("./Model_State/"+str(model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10") == False:
        os.mkdir("./Model_State/"+str(model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10")
    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    optimizer = optim.Adam(model.parameters(), lr=args.Adam_lr)
    # optimizer = optim.SGD(model.parameters(),lr=args.aT_Adam_lr,momentum=args.momentum)
    print(optimizer)

    train_acc_array = []
    # train_acc=0.0
    test_acc_array = []
    # test_acc=0.0
    for epoch in range(1, args.k_allTrain_epochs + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader, args)
        test_acc_array.append(test_acc)

    dir_model_state = "./Model_State/"+str(model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10"+"/"+str(
        model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10"+"_"+str(args.k_allTrain_epochs)+".pkl"
    torch.save(model.state_dict(), dir_model_state)

    layer_id = 0
    for child in model.children():
        layer_id += 1
        print("layer Id: "+str(layer_id), child)
        for param in child.parameters():
            param.requires_grad = False

    model.fc1.weight.requires_grad = True
    model.fc1.bias.requires_grad = True

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.Adam_lr*0.1)
    print(optimizer)

    for epoch in range(args.k_allTrain_epochs + 1, args.total_epochs + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader, args)
        test_acc_array.append(test_acc)

    return train_acc_array, test_acc_array
    # dirs = "./Result_npz/"
    # if not os.path.exists(dirs):
    #     os.mkdir(dirs)
    # # np.savez(dirs+"/acc"+str(int(microtrain_steps/display_step))+".npz", test_acc_array, train_acc_array)
    # np.savez(dirs+"/acc"+".npz", test_acc_array, train_acc_array)


def all_train(args, model, device):
    model = model.to(device)
    model.share_memory().to(device)
    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    train_acc_array = []
    # train_acc=0.0
    test_acc_array = []
    # test_acc=0.0

    init_lr = args.SGD_lr
    # optimizer = optim.SGD(model.parameters(),lr=args.aT_Adam_lr,momentum=args.momentum)

    # window counting
    w1 = args.LR_window1
    w12a = args.LR_window1+args.LR_window2
    w12b = args.LR_window1+args.LR_window2*2
    w123a = w12b+args.LR_window3
    w123b = w12b+args.LR_window3*2
    w123c = w12b+args.LR_window3*3
    w123d = w12b+args.LR_window3*4

    # lr window 1
    train_window(0, w1, args, model, device, train_loader,
                test_loader, init_lr, train_acc_array, test_acc_array)
    # lr window 2
    train_window(w1, w12a, args, model, device, train_loader,
                    test_loader, init_lr*0.5, train_acc_array, test_acc_array)
    train_window(w12a, w12b, args, model, device, train_loader,
                    test_loader, init_lr*0.1, train_acc_array, test_acc_array)
    # lr window 3
    train_window(w12b, w123a, args, model, device, train_loader,
                    test_loader, init_lr*0.05, train_acc_array, test_acc_array)
    train_window(w123a, w123b, args, model, device, train_loader,
                    test_loader, init_lr*0.01, train_acc_array, test_acc_array)
    train_window(w123b, w123c, args, model, device, train_loader,
                    test_loader, init_lr*0.005, train_acc_array, test_acc_array)
    train_window(w123c, w123d, args, model, device, train_loader,
                    test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    model_name = str(model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10"
    k_model_name = model_name+"_"+str(args.k_allTrain_epochs)
    dir_model_state = "../Model_State/"+model_name+"/"+k_model_name+".pkl"
    torch.save(model.state_dict(), dir_model_state)

    return train_acc_array, test_acc_array


def micro_train(args, model, device):
    model = model.to(device)
    model.share_memory().to(device)
    torch.manual_seed(args.seed)
#
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    train_acc_array = []
    # train_acc=0.0
    test_acc_array = []
    # test_acc=0.0

    init_lr = args.SGD_lr
    # optimizer = optim.SGD(model.parameters(),lr=args.aT_Adam_lr,momentum=args.momentum)

    w1 = args.LR_window1
    w12a = args.LR_window1+args.LR_window2
    w12b = args.LR_window1+args.LR_window2*2
    w123a = w12b+args.LR_window3
    w123b = w12b+args.LR_window3*2
    w123c = w12b+args.LR_window3*3
    w123d = w12b+args.LR_window3*4
    # lr windows 1
    if args.k_allTrain_epochs < w1:
        half_freeze(0, w1, args, model, device, train_loader,
                    test_loader, init_lr, train_acc_array, test_acc_array)
        train_window(w1, w12a, args, model, device, train_loader,
                     test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        train_window(w12a, w12b, args, model, device, train_loader,
                     test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        train_window(w12b, w123a, args, model, device, train_loader,
                     test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    # lr windows 2
    if w1 <= args.k_allTrain_epochs < w12a:
        train_window(0, w1, args, model, device, train_loader,
                     test_loader, init_lr, train_acc_array, test_acc_array)
        half_freeze(w1, w12a, args, model, device, train_loader,
                    test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        train_window(w12a, w12b, args, model, device, train_loader,
                     test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        train_window(w12b, w123a, args, model, device, train_loader,
                     test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    if w12a <= args.k_allTrain_epochs < w12b:
        train_window(0, w1, args, model, device, train_loader,
                     test_loader, init_lr, train_acc_array, test_acc_array)
        train_window(w1, w12a, args, model, device, train_loader,
                     test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        half_freeze(w12a, w12b, args, model, device, train_loader,
                    test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        train_window(w12b, w123a, args, model, device, train_loader,
                     test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    # lr windows 3
    if w12b <= args.k_allTrain_epochs < w123a:
        train_window(0, w1, args, model, device, train_loader,
                     test_loader, init_lr, train_acc_array, test_acc_array)
        train_window(w1, w12a, args, model, device, train_loader,
                     test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        train_window(w12a, w12b, args, model, device, train_loader,
                     test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        half_freeze(w12b, w123a, args, model, device, train_loader,
                    test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    if w123a <= args.k_allTrain_epochs < w123b:
        train_window(0, w1, args, model, device, train_loader,
                     test_loader, init_lr, train_acc_array, test_acc_array)
        train_window(w1, w12a, args, model, device, train_loader,
                     test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        train_window(w12a, w12b, args, model, device, train_loader,
                     test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        train_window(w12b, w123a, args, model, device, train_loader,
                    test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        half_freeze(w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    if w123b <= args.k_allTrain_epochs < w123c:
        train_window(0, w1, args, model, device, train_loader,
                     test_loader, init_lr, train_acc_array, test_acc_array)
        train_window(w1, w12a, args, model, device, train_loader,
                     test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        train_window(w12a, w12b, args, model, device, train_loader,
                     test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        train_window(w12b, w123a, args, model, device, train_loader,
                    test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        half_freeze(w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    if w123c <= args.k_allTrain_epochs < w123d:
        train_window(0, w1, args, model, device, train_loader,
                     test_loader, init_lr, train_acc_array, test_acc_array)
        train_window(w1, w12a, args, model, device, train_loader,
                     test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        train_window(w12a, w12b, args, model, device, train_loader,
                     test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        train_window(w12b, w123a, args, model, device, train_loader,
                    test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        half_freeze(w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    dir_model_state = "../Model_State/"+str(model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10"+"/"+str(
        model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10"+"_"+str(args.k_allTrain_epochs)+".pkl"
    torch.save(model.state_dict(), dir_model_state)

    return train_acc_array, test_acc_array


def half_freeze(L_W, R_W, args, model, device, train_loader, test_loader, lr, train_acc_array, test_acc_array):
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    print(optimizer)
    for epoch in range(L_W+1, args.k_allTrain_epochs + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader, args)
        test_acc_array.append(test_acc)

    layer_id = 0
    for child in model.children():
        layer_id += 1
        print("layer Id: "+str(layer_id), child)
        for param in child.parameters():
            param.requires_grad = False

    model.fc1.weight.requires_grad = True
    model.fc1.bias.requires_grad = True

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    print("---------> freeze the convolutional layer <---------")
    for epoch in range(args.k_allTrain_epochs + 1, R_W + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader, args)
        test_acc_array.append(test_acc)


def train_window(L_W, R_W, args, model, device, train_loader, test_loader, lr, train_acc_array, test_acc_array):
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    print("---------> Step into another training window <---------")
    print(optimizer)
    for epoch in range(L_W+1, R_W + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader, args)
        test_acc_array.append(test_acc)


def _train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss = loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
    # return train_loss


def _train_acc(epoch, model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
    test_loss /= len(data_loader.dataset)
    print('-----------------------------------------------------------------')
    print('Train Epoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(epoch,
                                                                                               test_loss, correct, len(
                                                                                                   data_loader.dataset),
                                                                                               100. * correct / len(data_loader.dataset)))
    # with open("./Result_npz/outputlog.txt", "a+") as f:
    #     print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    #     test_loss, correct, len(data_loader.dataset),
    #     100. * correct / len(data_loader.dataset)), file=f)
    acc = 100. * correct / len(data_loader.dataset)
    return acc


def _test_epoch(epoch, model, device, data_loader, args):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
    test_loss /= len(data_loader.dataset)
    print('Train Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(epoch,
                                                                                              test_loss, correct, len(
                                                                                                  data_loader.dataset),
                                                                                              100. * correct / len(data_loader.dataset)))
    print('-----------------------------------------------------------------')
    with open("../Result_npz/"+str(model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10"+"/TestLog_"+str(args.k_allTrain_epochs)+".txt", "a+") as f:
        print('Train Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(epoch,
                                                                                                  test_loss, correct, len(
                                                                                                      data_loader.dataset),
                                                                                                  100. * correct / len(data_loader.dataset)), file=f)
    return 100. * correct / len(data_loader.dataset)
