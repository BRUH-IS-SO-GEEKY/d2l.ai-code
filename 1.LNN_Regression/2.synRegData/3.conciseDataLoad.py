
import random
import math
import time
import numpy as np 
import torch
from torch import nn
from d2l import torch as d2l

class SyntheticRegressionData(d2l.DataModule) : #@save
    """Synthetic data for linear regression"""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
       batch_size=32):
      super().__init__()
      self.save_hyperparameters()
      n = num_train + num_val
      self.X = torch.randn(n, len(w))
      noise = torch.randn(n, 1) * noise
      self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise

data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)

print('features:', data.X[0],'\nlabel:', data.y[0])

@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset, self.batch_size,
                                       shuffle=train)

@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)


X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)

len(data.train_dataloader()) 
"""In built API framework provides with the len method"""
