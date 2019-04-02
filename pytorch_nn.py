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
        self.fc1 = nn.Linear(7*98,size_layer)
        self.fc1_dropout = nn.Dropout(self.p)
        self.fc2 = nn.Linear(size_layer,size_layer)
        self.fc2_dropout = nn.Dropout(self.p)
        self.fc3 = nn.Linear(size_layer,size_layer)
        self.fc3_dropout = nn.Dropout(self.p)
        self.fc4 = nn.Linear(size_layer,2)

        # self.conv1 = nn.Conv2d(1,8,4,2)
        # self.conv2 = nn.Conv2d(1,8,4,2)
    def forward(self,x):
        x = F.relu(self.fc1_dropout(self.fc1(x)))
        x = F.relu(self.fc2_dropout(self.fc2(x)))
        x = F.relu(self.fc3_dropout(self.fc3(x)))
        x = self.fc4(x)
        return x

def get_spikes_with_history(neural_data,bins_before,bins_after,bins_current=1):

    num_examples=neural_data.shape[0] #Number of total time bins we have neural data for
    num_neurons=neural_data.shape[1] #Number of neurons
    surrounding_bins=bins_before+bins_after+bins_current #Number of surrounding time bins used for prediction
    X=np.empty([num_examples,surrounding_bins,num_neurons]) #Initialize covariate matrix with NaNs
    X[:] = np.zeros_like(X)
    #Loop through each time bin, and collect the spikes occurring in surrounding time bins
    #Note that the first "bins_before" and last "bins_after" rows of X will remain filled with NaNs, since they don't get filled in below.
    #This is because, for example, we cannot collect 10 time bins of spikes before time bin 8
    start_idx=0
    for i in range(num_examples-bins_before-bins_after): #The first bins_before and last bins_after bins don't get filled in
        end_idx=start_idx+surrounding_bins; #The bins of neural data we will be including are between start_idx and end_idx (which will have length "surrounding_bins")
        X[i+bins_before,:,:]=neural_data[start_idx:end_idx,:] #Put neural data from surrounding bins in X, starting at row "bins_before"
        start_idx=start_idx+1;

    # X = X[6:X.shape[0]-7,:,:]
    return X

def data_loader(batch_size):
    data_features = pd.read_csv('monkey_features_300.csv')
    data_targets = pd.read_csv('monkey_targets_300.csv')
    data_features = data_features.iloc[:,:].as_matrix()
    X = get_spikes_with_history(data_features,6,0)

    X_mean=np.nanmean(X,axis=0)
    X_std=np.nanstd(X,axis=0)
    X=(X-X_mean)/X_std

    X = X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))

    trial_idx = data_targets.iloc[:,4].as_matrix()
    move_idx = data_targets.iloc[:,:5].as_matrix()

    data_targets = data_targets.iloc[:,:2].as_matrix()
    for idx in range(len(trial_idx)):
        if trial_idx[idx] == 600:
            break

    train = TensorDataset(torch.tensor(X[:idx,:]).float(), torch.tensor(data_targets[:idx,:]).float())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)

    valid = TensorDataset(torch.tensor(X[idx:,:]).float(), torch.tensor(data_targets[idx:,:]).float())
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, valid_loader


def main():
    n_epoch = 14
    batch_size = 128
    window_size = 15000/batch_size * n_epoch + 200
    size_layer = 512

    train_loader, valid_loader = data_loader(batch_size)

    network = Network(size_layer,0.4)
    optimiser = optim.Adam(network.parameters(), lr=1e-4, betas=(0.5, 0.999))

    log_loss = []
    log_loss_val = []
    log_it_val = []
    for epoch in range(n_epoch):
        for batch_idx, (neuron, position) in enumerate(train_loader):
            optimiser.zero_grad()
            output = network(neuron)
            loss = F.mse_loss(output, position)
            loss.backward()
            optimiser.step()
            log_loss.append(loss.detach().numpy())

            if batch_idx % 50 == 0 and batch_idx !=0:
                print('----- Epoch: ' + str(epoch+1) + ' Iteration: ' + str(batch_idx) + ' Loss: ' + str(log_loss[-1]) + '-----')
        loss_val = 0

        for batch_idx, (neuron, position) in enumerate(valid_loader):
            network.eval()
            output_v = network(neuron)
            loss_val += F.mse_loss(output_v, position)
        loss_val /= (batch_idx+1)
        log_loss_val.append(loss_val.detach().numpy())
        log_it_val.append((epoch+1)*len(log_loss_val)/batch_size)
        print('Validation loss: ',log_loss_val[-1])

        # plt.figure()
        plt.axis([0,window_size,0,1400])
        plt.plot(log_loss,color='b')
        plt.plot(log_it_val,log_loss_val,color='r')
        plt.draw()
        plt.pause(0.01)

    plt.show()
if __name__ == '__main__':
    main()
