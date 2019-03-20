import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset, TensorDataset
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

class Network(nn.Module):
    def __init__(self, size_layer,p):
        self.p = p
        super(Network,self).__init__()
        self.fc1 = nn.Linear(98,size_layer)
        self.fc1_dropout = nn.Dropout(self.p)
        self.fc2 = nn.Linear(size_layer,size_layer)
        self.fc2_dropout = nn.Dropout(self.p)
        self.fc3 = nn.Linear(size_layer,size_layer)
        self.fc3_dropout = nn.Dropout(self.p)
        self.fc4 = nn.Linear(size_layer,3)

        # self.conv1 = nn.Conv2d(1,8,4,2)
        # self.conv2 = nn.Conv2d(1,8,4,2)
    def forward(self,x):
        x = F.relu(self.fc1_dropout(self.fc1(x)))
        x = F.relu(self.fc2_dropout(self.fc2(x)))
        x = F.relu(self.fc3_dropout(self.fc3(x)))
        x = (self.fc4(x))

        # x.view(batch_size,98,-1,1)
        # x = self,conv1(x)
        return x

def data_loader(batch_size):
    data_features = pd.read_csv('monkey_features_classification.csv')
    data_targets = pd.read_csv('monkey_output_classification.csv')
    trial_idx = data_targets.iloc[:,2].as_matrix()
    move_idx = data_targets.iloc[:,1].as_matrix()
    data_features = data_features.iloc[:,:].as_matrix()
    for idx in range(len(trial_idx)):
        if trial_idx[idx] == 700:
            break

    train = TensorDataset(torch.tensor(data_features[:idx,:]).float(), torch.tensor(data_targets[:idx,:]).float())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)

    valid = TensorDataset(torch.tensor(data_features[idx:,:]).float(), torch.tensor(data_targets[idx:,:]).float())
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader


def main():
    n_epoch = 30
    batch_size = 128
    window_size = 25362/batch_size * n_epoch + 200
    size_layer = 512

    train_loader, valid_loader = data_loader(batch_size)

    network = Network(size_layer,0.5)
    optimiser = optim.Adam(network.parameters(), lr=1e-3, betas=(0.5, 0.999))

    log_loss = []
    log_loss_val = []
    log_it_val = []
    for epoch in range(n_epoch):
        for batch_idx, (neuron, position) in enumerate(train_loader):
            optimiser.zero_grad()
            output = network(neuron)
            loss = F.l1_loss(output, position)
            loss.backward()
            optimiser.step()
            log_loss.append(loss.detach().numpy())

            if batch_idx % 50 == 0 and batch_idx !=0:
                print('----- Epoch: ' + str(epoch+1) + ' Iteration: ' + str(batch_idx) + ' Loss: ' + str(log_loss[-1]) + '-----')
        loss_val = 0

        for batch_idx, (neuron, position) in enumerate(valid_loader):
            network.eval()
            output_v = network(neuron)
            loss_val += F.l1_loss(output_v, position)
        loss_val /= (batch_idx+1)
        log_loss_val.append(loss_val.detach().numpy())
        log_it_val.append((epoch+1)*25362/batch_size)

        # plt.figure()
        plt.axis([0,window_size,0,40])
        plt.plot(log_loss,color='b')
        plt.plot(log_it_val,log_loss_val,color='r')
        plt.draw()
        plt.pause(0.01)

    plt.show()
if __name__ == '__main__':
    main()
