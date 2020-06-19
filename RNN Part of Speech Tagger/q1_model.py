import torch
import torch.nn as nn


class SoftmaxModel(nn.Module):
    """
    Implements a Softmax classifier with cross-entropy loss.
    """

    def __init__(self, n_features: int, n_classes: int):
        """
        Contructs a SoftmaxModel.

        :param n_features: The size of model inputs
        :param n_classes: The size of model outputs
        """
        super(SoftmaxModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        # Problem 1a: Replace the following with your own code.
        w = torch.rand(n_features,n_classes)
        b = torch.ones(n_classes)

        # Add w and b to the model parameters and automatically set
        # requires_grad=True
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: An input tensor of shape (batch_size, n_features)
        :return: An output tensor of shape (batch_size, n_classes)
        """
        batch_size = x.size()[0]
        b = self.b.repeat(batch_size,1)
        w = self.w
        return x @ w + b