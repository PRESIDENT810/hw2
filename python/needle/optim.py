"""Optimization module"""
import needle as ndl
import numpy as np
from typing import Dict, Union, List


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    params: List[ndl.nn.Parameter]
    u: Dict[int, Union[int, ndl.Tensor]]

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        for i, param in enumerate(self.params):
            self.u[i] = 0
        self.weight_decay = weight_decay

    def step(self):
        for i, param in enumerate(self.params):
            # WHAT THE FUCK? Why it has to be momentum including decay
            grad = ndl.Tensor(param.grad, dtype=param.dtype) + self.weight_decay * param.data
            assert(grad.shape == param.shape)
            self.u[i] = (1 - self.momentum) * grad + self.momentum * self.u[i]
            self.params[i].data = param - self.lr * self.u[i]
            self.params[i].detach()
            self.u[i].detach()


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
