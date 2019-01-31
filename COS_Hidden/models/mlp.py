import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from data_process._data_process import plot_loss

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, std=0.015)

    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, std=0.015)
        # m.weight.data.normal_(0, 0.015)
        # m.bias.data.normal_(0, 0.015)

#---------------------------------
# Stochastic Multilayer Perceptron
#---------------------------------


class MLP(nn.Module):
    def __init__(self, Input_dim=1, Output_dim=1, Hidden_size=100, Epoch=4000, Optim_method='SGD', print_interval=50, plot_=False):
        super(BaseModel, self).__init__()
        self.input_dim = Input_dim
        self.output_dim = Output_dim
        self.hidden_size = Hidden_size
        self.epoch = Epoch
        self.optim = Optim_method
        self.weight_IH = None
        self.bias_IH = None
        self.weight_HO = None
        self.bias_HO = None
        # self.ridge_alpha = Ridge_alpha
        # self.regressor = Ridge(alpha=self.ridge_alpha)

        self.Print_interval = print_interval
        
        self.plot_ = plot_

        #
        self.IH = nn.Linear(self.input_dim, self.hidden_size)
        self.HO = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, input):
        hidden = self.IH(input)
        output = F.relu(hidden)
        self.weight_IH = self.IH.weight.data.clone()
        output = self.HO(output)
        self.weight_HO = self.HO.weight.data.clone()

        return F.log_softmax(output,dim=1) 

    def fit(self, input, target,save_road='./Results/'):
        print('========start training========')
        criterion = F.nll_loss()
        if self.Optim_method == 'SGD':
            optimizer = optim.SGD(
                self.parameters(), lr=self.Learn_rate, momentum=0.99)
        if self.Optim_method == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.Learn_rate)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.95)

        if self.plot_ == True:
            plot_losses = []
        # train_print_loss_total = 0  # Reset every print_every
        train_plot_loss_total = 0  # Reset every plot_every

        # plt.figure(1,figsize=(30,5))# continuously plot
        # plt.ion() # continuously plot

        # input_size=input.size(0)# continuously plot
        # time_period=np.arange(input_size)# continuously plot

        # begin to train
        for iter in range(1, self.epoch + 1):
            scheduler.step()
            # input: shape[batch,input_dim]
            # prediction: shape[batch,output_dim=1]
            output = self.forward(input)
            loss = criterion(output, target, size_average=False).item()
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct = pred.eq(target).sum().item()
            avarage_loss  = loss / input.data.size(0)
            
            print('Train Epoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(epoch,
                                                                                                       avarage_loss, correct, input.data.size(0), 100. * correct / input.data.size(0)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # target_view=input[:,-1,:].data.numpy()# continuously plot
            # prediction_view=prediction[:,-1,:].data.numpy()
            # plt.plot(time_period,target_view.flatten(),'r-')
            # plt.plot(time_period,prediction_view.flatten(),'b-')
            # plt.draw();plt.pause(0.05)

            if iter % self.Print_interval == 0:
                print('Train Epoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(epoch,
                                                                                                        avarage_loss, correct, input.data.size(0), 100. * correct / input.data.size(0)))
            if self.plot_ == True:
                    plot_losses.append(avarage_loss)

        if self.plot_ == True:
            plot_loss(plot_losses, Fig_name=save_road+'Loss_MLP' +
                  '_H'+str(self.hidden_size)+'_I'+str(self.epoch)+'_'+self.optim)