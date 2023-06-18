"""The module.
"""
from typing import List, Callable, Any, Optional
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    in_features: int
    out_features: int
    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights using Kaiming initialization
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        y = X @ self.weight
        if self.bias is None:
            return y
        return y + self.bias.broadcast_to(y.shape)


class Flatten(Module):
    def forward(self, X):
        shape = X.shape
        if len(shape) == 1:
            return X
        total = 1
        for dim in X.shape:
            total *= dim
        return ops.reshape(X, (shape[0], int(total / shape[0])))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for mod in self.modules:
            x = mod(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        loss = ops.logsumexp(logits, axes=1)
        y_onehot = logits * init.one_hot(logits.shape[1], y)
        return (loss - y_onehot.sum(axes=1)).sum() / logits.shape[0]


class BatchNorm1d(Module):
    weight: Parameter
    bias: Parameter
    running_mean: ops.NDArray
    running_var: ops.NDArray

    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(1, dim, requires_grad=True))
        self.bias = Parameter(init.zeros(1, dim, requires_grad=True))
        self.running_mean = init.zeros(1, dim, requires_grad=True).reshape(dim)
        self.running_var = init.ones(1, dim, requires_grad=True).reshape(dim)

    def forward(self, x: Tensor) -> Tensor:
        mean = ops.summation(x, axes=0)
        mean = mean / x.shape[0]
        # BatchNorm uses the running estimates of mean and variance instead of batch statistics at test time
        mean = mean.reshape((1, mean.shape[0]))
        var = x - ops.broadcast_to(mean, x.shape)
        var = ops.power_scalar(var, 2)
        var = ops.summation(var, 0)
        var = var.reshape((1, var.shape[0]))
        var = var / x.shape[0]
        running_mean = self.running_mean.reshape((1, self.dim))
        running_var = self.running_var.reshape((1, self.dim))

        if self.training:  # Training phase
            running_mean = (running_mean * (1 - self.momentum) + mean * self.momentum)
            running_var = (running_var * (1 - self.momentum) + var * self.momentum)
        else:  # Testing phase
            mean = self.running_mean
            var = self.running_var
        self.running_mean = running_mean.reshape(self.dim).detach()
        self.running_var = running_var.reshape(self.dim).detach()
        var = var + self.eps
        var = ops.power_scalar(var, 0.5)
        y = x - ops.broadcast_to(mean, x.shape)
        y = y / ops.broadcast_to(var, x.shape)
        y = y * self.weight.broadcast_to(x.shape)
        y = y + self.bias.broadcast_to(y.shape)
        return y


class LayerNorm1d(Module):
    weight: Parameter
    bias: Parameter
    dim: int
    eps: float

    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(1, dim, requires_grad=True))
        self.bias = Parameter(init.zeros(1, dim, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        mean = ops.summation(x, axes=1)
        mean = mean / x.shape[1]
        mean = mean.reshape((mean.shape[0], 1))
        mean = ops.broadcast_to(mean, x.shape)
        var = x - mean
        var = ops.power_scalar(var, 2)
        var = ops.summation(var, 1)
        var = var.reshape((var.shape[0], 1))
        var = var / x.shape[1]
        var = var + self.eps
        var = ops.power_scalar(var, 0.5)
        var = ops.broadcast_to(var, x.shape)
        y = x - mean
        y = y / var
        y = y * self.weight.broadcast_to(x.shape)
        y = y + self.bias.broadcast_to(y.shape)
        return y


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:  # Testing phase
            return x
        else:  # Training phase
            mask = init.randb(*x.shape, p=(1 - self.p)) / (1 - self.p)
            return x * mask


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
