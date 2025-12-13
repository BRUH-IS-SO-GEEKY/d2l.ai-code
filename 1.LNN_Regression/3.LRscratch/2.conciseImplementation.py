
import numpy as np 
import torch 
from torch import nn
from d2l import torch as d2l

"""Defining the Model"""
#Fully connected layer is defined in Linear and Lazylinear classes

class LinearRegression(d2l.Moudule) :
    """The linear regression models implemented with high-level APIs"""
    def __init__(self, lr) :
        super.__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

"""In the forward method we just invoke the built-in __call__ method of the predefined 
layers to compute the ouptput"""

@d2l.add_to_class(LinearRegression) 
def forward(self , X) :
    return self.net(X)

"""Defining the loss function"""
@d2l.add_to_class(LinearRegression) 
def loss(self, y_hat, y) :
    fn = nn.MSELoss()
    return fn(y_hat, y)

"""Defining the optimization algorithm"""
@d2l.add_to_class(LinearRegression)
def configure_optimizers(self) :
    return torch.optim.SGD(self.parameters(), self.lr)

"""Get parameters"""
@d2l.add_to_class(LinearRegression)
def get_w_b(self) :
    return (self.net.weight.data, self.net.bias.data)

"""Training"""

model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)

w,b= model.get_w_b()

print(f"error in estimating w : {data.w - w.reshape(data.w.shape)}")
print(f"error in estimating b : {data.b - b}")

