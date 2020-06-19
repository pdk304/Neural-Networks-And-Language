import numpy as np


def loss(y_hat: np.ndarray, y: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """
    Computes the cross-entropy loss of a batch of outputs.

    :param y_hat: The outputs produced by the model
    :param y: The correct outputs from the training dataset
    :param eps: A stabilizing term
    :return: The total loss for this batch
    """
    return -1. * np.sum(np.log(y_hat[range(len(y)), y] + eps))


def loss_grad(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Implements the computation of delta_o, the 
    gradient of the loss function with respect to z_o. 
   
    :param y_hat: The outputs produced by the model
    :param y: The correct outputs from the training dataset
    :return: delta_o for this batch. delta_o should be a matrix of 
        shape (batch_size, vocab_size), where delta_o[i, j] is
        dL_CE/dz_o_j, computed for the ith training example in the
        batch.
    """
    y_output = np.zeros((y.size, 250))
    y_output[np.arange(y.size), y] = 1
    delta_o = y_hat - y_output
    return delta_o