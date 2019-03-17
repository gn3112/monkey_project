import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset, TensorDataset
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.fc1 = nn.Linear(98,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,3)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x)) * 120
        return x
def main():
    data_features = pd.read_csv('monkey_features.csv')
    data_targets = pd.read_csv('monkey_targets.csv')
    data_targets = data_targets.iloc[:,:3].as_matrix()
    data_features = data_features.iloc[:,:].as_matrix()
    train = TensorDataset(torch.tensor(data_features).float(), torch.tensor(data_targets).float())
    train_loader = DataLoader(train, batch_size=128, shuffle=True)

    network = Network()
    optimiser = optim.Adam(network.parameters(), lr=1e-2, betas=(0.5, 0.999))

    log_loss = []
    for epoch in range(20):
        for batch_idx, (neuron, position) in enumerate(train_loader):
            optimiser.zero_grad()
            output = network(neuron)
            loss = F.l1_loss(output, position)
            loss.backward()
            optimiser.step()
            log_loss.append(loss.detach().numpy())

            if batch_idx % 100 == 0 and batch_idx !=0:
                print('----- Epoch: ' + str(epoch+1) + ' Iteration: ' + str(batch_idx) + ' Loss: ' + str(log_loss[-1]) + '-----')


if __name__ == '__main__':
    main()
