import pickle
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np

from cross_entropy import loss, loss_grad
from model import Model, softmax


def relative_error(a, b):
    return np.abs(a - b) / (np.abs(a) + np.abs(b))


def check_delta_o(model: Model, x: np.ndarray, y: np.ndarray,
                  eps: float = 1e-4, bound: float = 1e-4, num_trials: int =
                  10) -> bool:
    """
    Checks the delta_o computed for a model against a numerically
    estimated approximation for delta_o.

    :param model: A model
    :param x: An input batch
    :param y: The correct outputs of the batch
    :param eps: The amount by which to perturb z_o
    :param bound: The maximum allowable error
    :param num_trials: The number of times to repeat the estimation
    :return: True if the model's delta_o is approximately equal to the
        estimated value, False otherwise
    """
    batch_size = len(x)
    z = np.random.normal(size=(batch_size, model.vocab_size))
    y_hat = softmax(z)
    delta_o = loss_grad(y_hat, y)

    # Make sure dimensions match
    delta_o_shape = (batch_size, model.vocab_size)
    if delta_o.shape != delta_o_shape:
        msg = "delta_o's shape is {} instead of {}".format(delta_o_shape,
                                                           delta_o.shape)
        raise RuntimeError(msg)

    # Numerically estimate the gradient
    for _ in range(num_trials):

        i = np.random.randint(0, delta_o_shape[0])
        j = np.random.randint(0, delta_o_shape[1])

        # Perturb z[i, j] by a little bit
        z_plus = z.copy()
        z_plus[i, j] += eps
        loss_plus = loss(softmax(z_plus), y)

        z_minus = z.copy()
        z_minus[i, j] -= eps
        loss_minus = loss(softmax(z_minus), y)

        # Compute the relative error
        delta_o_ij_estimate = (loss_plus - loss_minus) / (2. * eps)
        error = relative_error(delta_o_ij_estimate, delta_o[i, j])
        if error > bound:
            print("The relative error of your delta_o is", error)
            return False

    print("Your delta_o looks okay.")
    return True


def check_param_grad(model: Model, param_name: str, x: np.ndarray,
                     y: np.ndarray, eps: float = 1e-4, bound: float = 1e-4,
                     num_trials: int = 1000) -> bool:
    """
    Checks the gradient of the loss with respect to a parameter computed
    by a model using Model.backward() against a numerically estimated
    approximation.

    :param model: A model
    :param param_name: The parameter whose gradient will be checked
    :param x: An input batch
    :param y: The correct outputs of the batch
    :param eps: The amount by which to perturb the parameter value
    :param bound: The maximum allowable error
    :param num_trials: The number of times to repeat the estimation
    :return: True if the model's computed gradient is approximately
        equal to the estimated value, False otherwise
    """
    y_hat = model.forward(x)
    delta_o = loss_grad(y_hat, y)
    model.backward(delta_o)

    # Make sure dimensions match
    grad_shape = model.grads[param_name].shape
    if grad_shape != model.params[param_name].shape:
        print("Your", param_name, "gradient's shape is", grad_shape,
              "instead of", model.params[param_name].shape)
        return False

    # Numerically estimate the gradient
    for _ in range(num_trials):
        slc = tuple([np.random.randint(0, i) for i in grad_shape])

        # Perturb the weights by a little bit
        model_plus = deepcopy(model)
        model_plus.params[param_name][slc] += eps
        loss_plus = loss(model_plus.forward(x), y)

        model_minus = deepcopy(model)
        model_minus.params[param_name][slc] -= eps
        loss_minus = loss(model_minus.forward(x), y)

        # Compute the relative error
        grad_estimate = (loss_plus - loss_minus) / (2. * eps)
        error = relative_error(grad_estimate, model.grads[param_name][slc])
        if error > bound:
            print("The relative error of your gradient for", param_name, "is",
                  error)
            return False

    print("Your gradient for", param_name, "looks okay.")
    return True


def check_gradients(filename: str = "partially_trained.pk", seed: int = None):
    """
    Checks all gradients computed by a saved model.

    :param filename: The name of the saved model
    :param seed: A seed for the random number generator
    :return: None
    """
    if seed is not None:
        np.random.seed(seed)  # Originally set to 0
    np.seterr(all="ignore")  # Supress a harmless warning

    model = Model.from_file(filename)
    with open("data.pk", "rb") as f:
        data = pickle.load(f)

    train_x, train_y = data["train_inputs"], data["train_targets"]
    x = train_x[:100]
    y = train_y[:100]

    if not check_delta_o(model, x, y):
        return
    for param_name in model.params:
        check_param_grad(model, param_name, x, y)


def print_gradients(*delta_o_idxs: Tuple[int, int],
                    filename: str = "partially_trained.pk",
                    **grad_idxs: List[Union[Tuple[int, int], int]]):
    """
    Prints some of the gradients for grading.

    :param filename: The filename of a saved model
    :param delta_o_idxs: The delta_o indices to print
    :param grad_idxs: The indices of Model.grad to print
    :return: None
    """
    model = Model.from_file(filename)
    with open("data.pk", "rb") as f:
        data = pickle.load(f)
    train_x, train_y = data["train_inputs"], data["train_targets"]
    x = train_x[:100]
    y = train_y[:100]
 
    y_hat = model.forward(x)
    delta_o = loss_grad(y_hat, y)
    model.backward(delta_o)

    for idxs in delta_o_idxs:
        print("delta_o[{}]: {}".format(idxs, delta_o[idxs]))
    print()
    for param_name in grad_idxs:
        for idxs in grad_idxs[param_name]:
            print("{}_grad[{}]: {}".format(param_name, idxs,
                                           model.grads[param_name][idxs]))
        print()


if __name__ == "__main__":
    print_gradients((2, 5), (2, 121), (5, 33), (5, 31),
                     w_e=[(27, 2), (43, 3), (22, 4), (2, 5)],
                     w_h=[(10, 2), (15, 3), (30, 9), (35, 21)],
                     b_h=[10, 20], b_o=[0, 1, 2, 3])
    
