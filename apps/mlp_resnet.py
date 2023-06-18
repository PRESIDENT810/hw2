import sys

sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

from typing import Optional

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU()
    )


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for i in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)


def epoch(dataloader: ndl.data.DataLoader, model: ndl.nn.Module, opt: Optional[ndl.optim.Optimizer] = None):
    np.random.seed(4)

    losses = []
    hit = 0
    total = 0
    if opt is None:
        model.eval()
    else:
        model.train()

    for batch in iter(dataloader):
        X, y = batch
        logits = model(X)
        loss = ndl.nn.SoftmaxLoss().forward(logits, y)
        losses.append(loss.cached_data)
        loss.backward()
        # Update weights
        if opt is not None:
            opt.step()
        hit += (y.cached_data == X.cached_data.argmax(axis=1)).sum()
        total += y.shape[0]
    return np.average(np.array(losses)), hit / total


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
