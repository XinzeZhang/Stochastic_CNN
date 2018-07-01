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

class SCSF(nn.Module):
    def __init__(self,n_kernel=128):
        super(SCSF, self).__init__()
        self.n_kernel=n_kernel
        self.conv1 = nn.Conv2d(3, self.n_kernel, kernel_size=5,stride=1,padding=2)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(self.n_kernel*16*16, 10)

    def forward(self, x):
        # stride – the stride of the window. Default value is kernel_size, thus it is 2 here.
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2,stride=2))
        x = self.conv2_drop(x)
        x = x.view(-1, self.n_kernel*16*16)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)


