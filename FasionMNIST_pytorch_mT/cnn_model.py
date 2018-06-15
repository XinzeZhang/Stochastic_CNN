import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.normal_(m.weight.data,std=0.015)

    if isinstance(m,nn.Linear):
        m.weight.data.normal_(0, 0.015)
        m.bias.data.normal_(0, 0.015)
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
        # nn.init.kaiming_normal_(m.weight.data)
    #     # m.weight.data.normal_(0.0, 0.01)
    # elif classname.find('Linear') != -1:
    #     m.weight.data.normal_(0, 0.1)

class SCSF_Net(nn.Module):
    def __init__(self):
        super(SCSF_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5,stride=1,padding=2)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(12544, 10)

    def forward(self, x):
        # stride – the stride of the window. Default value is kernel_size, thus it is 2 here.
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2,stride=2))
        x = self.conv2_drop(x)
        x = x.view(-1, 12544)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)

class DCDF_Net(nn.Module):
    def __init__(self):
        super(DCDF_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # stride (the stride of the window) : Default value is kernel_size, thus it is 2 here.
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # stride (the stride of the window) : Default value is kernel_size, thus it is 2 here.
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)