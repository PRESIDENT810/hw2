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
    wrong = 0
    total = 0
    if opt is None:
        model.eval()
    else:
        model.train()

    for batch in iter(dataloader):
        X, y = batch
        output = model(X)
        loss = ndl.nn.SoftmaxLoss().forward(output, y)
        losses.append(loss.cached_data)
        # Update weights
        if opt is not None:
            loss.backward()
            opt.step()
        wrong += (y.numpy() != output.numpy().argmax(axis=1)).sum()
        total += y.shape[0]
    return wrong / total, np.average(np.array(losses))


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    train_dataset = ndl.data.MNISTDataset(
        data_dir + '/train-images-idx3-ubyte.gz',
        data_dir + '/train-labels-idx1-ubyte.gz'
    )
    test_dataset = ndl.data.MNISTDataset(
        data_dir + '/t10k-images-idx3-ubyte.gz',
        data_dir + '/t10k-labels-idx1-ubyte.gz',
    )
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size)
    model = MLPResNet(28 * 28 * 1, hidden_dim=hidden_dim)

    train_acc, train_avg_loss, test_acc, test_avg_loss = None, None, None, None

    # Train ResNet
    if optimizer is not None:
        opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)
        for e in range(epochs):
            train_acc, train_avg_loss = epoch(train_dataloader, model, opt)
            # print("train_acc={}, train_avg_loss={}".format(train_acc, train_avg_loss))

    # Test ResNet
    for e in range(epochs):
        test_acc, test_avg_loss = epoch(test_dataloader, model)

    return train_acc, train_avg_loss, test_acc, test_avg_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
