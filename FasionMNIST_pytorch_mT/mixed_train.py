import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np


def train(args, model, device):
    model = model.to(device)
    model.share_memory().to(device)
    if os.path.exists("./Model_State/"+str(model.__class__.__name__)) == False:
        os.mkdir("./Model_State/"+str(model.__class__.__name__))
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

    optimizer = optim.Adam(model.parameters(), lr=args.aT_Adam_lr)
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
        test_acc = _test_epoch(epoch, model, device, test_loader)
        test_acc_array.append(test_acc)

    dir_model_state = "./Model_State/"+str(model.__class__.__name__)+"/"+str(
        model.__class__.__name__)+"_"+str(args.k_allTrain_epochs)+".pkl"
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
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.aT_Adam_lr)
    print(optimizer)

    for epoch in range(1, args.total_epochs-args.k_allTrain_epochs-100 + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader)
        test_acc_array.append(test_acc)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.mT_Adam_lr)
    print(optimizer)

    for epoch in range(args.microTrain_epochs-100 + 1, args.microTrain_epochs+1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader)
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
    if os.path.exists("./Model_State/"+str(model.__class__.__name__)) == False:
        os.mkdir("./Model_State/"+str(model.__class__.__name__))
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

    train_acc_array = []
    # train_acc=0.0
    test_acc_array = []
    # test_acc=0.0

    init_lr = args.Adam_lr
    # optimizer = optim.SGD(model.parameters(),lr=args.aT_Adam_lr,momentum=args.momentum)

    # lr windows 1
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.0005,amsgrad=True)
    print(optimizer)
    for epoch in range(1, args.LR_window1 + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader)
        test_acc_array.append(test_acc)

    # lr windows 2
    optimizer = optim.Adam(model.parameters(), lr=init_lr*0.5, weight_decay=0.0005,amsgrad=True)
    print(optimizer)
    for epoch in range(1, args.LR_window2 + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader)
        test_acc_array.append(test_acc)

    optimizer = optim.Adam(model.parameters(), lr=init_lr*0.1, weight_decay=0.0005,amsgrad=True)
    print(optimizer)
    for epoch in range(1, args.LR_window2 + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader)
        test_acc_array.append(test_acc)

    # lr windows 3
    if args.LR_window3 > 0:
        optimizer = optim.Adam(
            model.parameters(), lr=init_lr*0.05, weight_decay=0.0005,amsgrad=True)
        print(optimizer)
        for epoch in range(1, args.LR_window3 + 1):
            _train_epoch(epoch, args, model, device, train_loader, optimizer)
            train_acc = _train_acc(epoch, model, device, train_loader)
            train_acc_array.append(train_acc)
            test_acc = _test_epoch(epoch, model, device, test_loader)
            test_acc_array.append(test_acc)

        optimizer = optim.Adam(
            model.parameters(), lr=init_lr*0.01, weight_decay=0.0005,amsgrad=True)
        print(optimizer)
        for epoch in range(1, args.LR_window3 + 1):
            _train_epoch(epoch, args, model, device, train_loader, optimizer)
            train_acc = _train_acc(epoch, model, device, train_loader)
            train_acc_array.append(train_acc)
            test_acc = _test_epoch(epoch, model, device, test_loader)
            test_acc_array.append(test_acc)

        optimizer = optim.Adam(
            model.parameters(), lr=init_lr*0.005, weight_decay=0.0005,amsgrad=True)
        print(optimizer)
        for epoch in range(1, args.LR_window3 + 1):
            _train_epoch(epoch, args, model, device, train_loader, optimizer)
            train_acc = _train_acc(epoch, model, device, train_loader)
            train_acc_array.append(train_acc)
            test_acc = _test_epoch(epoch, model, device, test_loader)
            test_acc_array.append(test_acc)

        optimizer = optim.Adam(
            model.parameters(), lr=init_lr*0.001, weight_decay=0.0005,amsgrad=True)
        print(optimizer)
        for epoch in range(1, args.LR_window3 + 1):
            _train_epoch(epoch, args, model, device, train_loader, optimizer)
            train_acc = _train_acc(epoch, model, device, train_loader)
            train_acc_array.append(train_acc)
            test_acc = _test_epoch(epoch, model, device, test_loader)
            test_acc_array.append(test_acc)

    dir_model_state = "./Model_State/"+str(model.__class__.__name__)+"/"+str(
        model.__class__.__name__)+"_"+str(args.k_allTrain_epochs)+".pkl"
    torch.save(model.state_dict(), dir_model_state)

    return train_acc_array, test_acc_array


def micro_train(args, model, device):
    model = model.to(device)
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

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.aT_Adam_lr)
    print(optimizer)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                       lr=args.SGD_lr, momentum=args.momentum)
    train_acc_array = []
    # train_acc=0.0
    test_acc_array = []
    # test_acc=0.0
    for epoch in range(1, args.total_epochs-args.k_allTrain_epochs-100 + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader)
        test_acc_array.append(test_acc)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.mT_Adam_lr)
    print(optimizer)
    for epoch in range(args.microTrain_epochs-100 + 1, args.microTrain_epochs+1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader)
        test_acc_array.append(test_acc)

    return train_acc_array, test_acc_array
    # dirs = "./Result_npz/"
    # if not os.path.exists(dirs):
    #     os.mkdir(dirs)
    # # np.savez(dirs+"/acc"+str(int(microtrain_steps/display_step))+".npz", test_acc_array, train_acc_array)
    # np.savez(dirs+"/acc"+".npz", test_acc_array, train_acc_array)


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


def _test_epoch(epoch, model, device, data_loader):
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
    with open("./Result_npz/outputlog.txt", "a+") as f:
        print('Train Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(epoch,
                                                                                                  test_loss, correct, len(
                                                                                                      data_loader.dataset),
                                                                                                  100. * correct / len(data_loader.dataset)), file=f)
    return 100. * correct / len(data_loader.dataset)
