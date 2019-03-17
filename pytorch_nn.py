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
    def __init__(self, size_layer):
        super(Network,self).__init__()
        self.fc1 = nn.Linear(98,size_layer)
        self.fc2 = nn.Linear(size_layer,size_layer)
        self.fc3 = nn.Linear(size_layer,3)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.tanh(self.fc3(x)) * 120
        x = self.fc3(x)
        return x
def main():
    n_epoch = 20
    batch_size = 256
    window_size = 25000/batch_size * n_epoch + 300
    size_layer = 512

    data_features = pd.read_csv('monkey_features.csv')
    data_targets = pd.read_csv('monkey_targets.csv')
    data_targets = data_targets.iloc[:,:3].as_matrix()
    data_features = data_features.iloc[:,:].as_matrix()
    train = TensorDataset(torch.tensor(data_features).float(), torch.tensor(data_targets).float())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    network = Network(size_layer)
    optimiser = optim.Adam(network.parameters(), lr=1e-2, betas=(0.5, 0.999))

    log_loss = []
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
        # plt.figure()
        plt.axis([0,window_size,0,25])
        plt.plot(log_loss,color='b')
        plt.draw()
        plt.pause(0.01)

    plt.show()
if __name__ == '__main__':
    main()
