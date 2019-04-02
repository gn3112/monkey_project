import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from torch import nn, optim
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from pytorch_nn import Network
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, Subset, TensorDataset
import torch


data_features = pd.read_csv('monkey_features.csv')
data_targets = pd.read_csv('monkey_targets.csv')
trial_idx = data_targets.iloc[:,4].as_matrix()
move_idx = data_targets.iloc[:,:5].as_matrix()
data_targets = data_targets.iloc[:,:3].as_matrix()
data_features = data_features.iloc[:,:].as_matrix()
for idx in range(len(trial_idx)):
    if trial_idx[idx] == 700:
        break

train = TensorDataset(torch.tensor(data_features[:idx,:]).float(), torch.tensor(data_targets[:idx,:]).float())
train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=4)

valid = TensorDataset(torch.tensor(data_features[idx:,:]).float(), torch.tensor(data_targets[idx:,:]).float())
valid_loader = DataLoader(valid, batch_size=128, shuffle=True, num_workers=4)

net = NeuralNetRegressor(
    Network,
    optimizer=torch.optim.Adam,
    # iterator_train = train_loader,
    # iterator_valid = valid_loader,
    module__size_layer = 512,
    module__p = 0.2,
    max_epochs=20,
    lr=0.0015,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)
net.initialize()
params = {
    'lr': [1e-3, 2e-3, 5e-3],
    'max_epochs': [15,25],
    'module__size_layer': [256,512,684],
    'module__p': [0.1,0.3,0.5]
}

gs = GridSearchCV(net, params, refit=False, cv=3,scoring='neg_mean_squared_error')
print(np.min(np.float32(data_features)))
gs.fit(np.float32(data_features),np.float32(data_targets))
print(gs.best_score_, gs.best_params_)
