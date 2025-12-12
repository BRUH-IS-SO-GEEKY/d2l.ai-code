
import math
import time
import numpy as np 
import torch
from d2l import torch as d2l


def add_to_class(Class) :
    """Register functions as methods in created class"""
    def wrapper(obj) :
        setattr(Class, obj.__name__, obj)
    return wrapper

class A :
    def __init__(self) :
        self.b = 1 

a = A()


@add_to_class(A) 
def do(self) :
    print(f"class atrribute b is {self.b}")

a.do()


class HyperParameters :
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented


class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print(f"self.a = {self.a} and self.b = {self.b}")
        print(f"There is no self.c = {not hasattr(self, 'c')}")
    
b = B(a=1, b=2, c=3)


class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()
    def draw(self, x, y, label, every_n=1):
        raise NotImplemented

board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)





        




