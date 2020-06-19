import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from q1_model import SoftmaxModel
from torch.utils.data import TensorDataset, DataLoader


def run_epoch(model: SoftmaxModel, loader: DataLoader,
              criterion: nn.Module, optimizer: optim.Optimizer) -> \
        torch.Tensor:
    """
    Trains the model for one epoch.

    :param model: The model to be trained
    :param loader: The DataLoader containing the data
    :param criterion: The loss function
    :param optimizer: The optimizer
    :return: Average minibatch loss of model on the epoch
    """

    total_loss = 0.
    for batch_idx, (x, y) in enumerate(loader):
        # forward pass
        y_hat = model(x)
        loss = 0.
        loss = criterion(y_hat,y)

        # backward pass
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # keep track of progress
        total_loss += loss

    return total_loss / batch_idx


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
          criterion: nn.Module, optimizer: optim.Optimizer, batch_size: int,
          epochs: int) -> List[torch.Tensor]:
    """
    Fits a model on the provided data.

    :param model: A classifier
    :param x: The model input, of shape (n_samples, n_features)
    :param y: The correct model output, of shape (n_samples,)
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param batch_size: The number of examples in each mini-batch
    :param epochs: The number of epochs to run
    :return: The loss of each epoch
    """
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size)
    losses = []
    for epoch in range(epochs):
        start_time = time.time()
        average_loss = run_epoch(model, loader, criterion, optimizer)
        duration = (time.time() - start_time)

        # Print status to stdout
        print("Epoch %d: loss = %.2f (%.3f sec)"
              % (epoch, average_loss, duration))
        losses.append(average_loss)

    return losses


def train_and_test_softmax_model(n_features: int = 1000, n_classes: int = 3,
                                 n_samples: int = 100, batch_size: int = 5,
                                 epochs: int = 40, lr: float = .01):
    """
    Trains a softmax model for a number of steps on random data.

    :param n_features: The size of input vectors
    :param n_classes: The size of output vectors
    :param n_samples: The number of examples in the training dataset
    :param batch_size: The number of examples in each mini-batch
    :param epochs: The number of epochs to train the model for
    :param lr: The learning rate to use for gradient descent
    :return: None
    """
    x = torch.rand(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    losses = []


    # training
    model = SoftmaxModel(n_features, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr)
    train(model,x,y,criterion,optimizer,batch_size,epochs)

    # testing
    for _ in range(batch_size):
        y_hat = model(x)
        losses.append(criterion(y_hat,y))

    # If your model is implemented correctly, the average loss should
    # fall close to zero rapidly.
    assert losses[-1] < .5
    print("Basic (non-exhaustive) classifier tests pass\n")


if __name__ == "__main__":
    train_and_test_softmax_model()