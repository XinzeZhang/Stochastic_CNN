import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np


def train(args, model, device):
    model = model.to(device)
    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    optimizer = optim.Adam(model.parameters(),
                          lr=args.Adam_lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.Adam_lr)

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

    return train_acc_array,test_acc_array
    # dirs = "./Result_npz/"
    # if not os.path.exists(dirs):
    #     os.mkdir(dirs)
    # # np.savez(dirs+"/acc"+str(int(microtrain_steps/display_step))+".npz", test_acc_array, train_acc_array)
    # np.savez(dirs+"/acc"+".npz", test_acc_array, train_acc_array)


def micro_train(args, model, device):
    model = model.to(device)
    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.SGD_lr, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=args.Adam_lr)

    train_acc_array = []
    # train_acc=0.0
    test_acc_array = []
    # test_acc=0.0
    for epoch in range(1, args.microTrain_epochs + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(epoch, model, device, test_loader)
        test_acc_array.append(test_acc)
    
    return train_acc_array,test_acc_array
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
        # if batch_idx % args.log_interval == 0:
        #     print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         pid, epoch, batch_idx * len(data), len(data_loader.dataset),
        #         100. * batch_idx / len(data_loader), loss.item()))
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
    print('Train Epoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(epoch,
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
    print('Train Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(epoch,
                                                                                                test_loss, correct, len(
                                                                                                    data_loader.dataset),
                                                                                                100. * correct / len(data_loader.dataset)))
    print('-----------------------------------------------------------------')
    with open("./Result_npz/outputlog.txt", "a+") as f:
        print('Train Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(epoch,
                                                                                                    test_loss, correct, len(
                                                                                                        data_loader.dataset),
                                                                                                    100. * correct / len(data_loader.dataset)), file=f)
    return 100. * correct / len(data_loader.dataset)
