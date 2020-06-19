import pickle
from typing import List

import numpy as np

from array_dict import ArrayDict


def zero_params(vocab_size: int, context_len: int, embedding_dim: int,
                hidden_dim: int) -> ArrayDict:
    """
    Constructs an ArrayDict of model parameters initialized to all
    zeros.

    :param vocab_size: The number of possible words in the vocabulary
    :param context_len: The number of input units
    :param embedding_dim: The size of the word embeddings
    :param hidden_dim: The number of hidden units
    :return: The initialized parameters
    """
    w_e = np.zeros((vocab_size, embedding_dim))
    w_h = np.zeros((hidden_dim, context_len * embedding_dim))
    w_o = np.zeros((vocab_size, hidden_dim))
    b_h = np.zeros(hidden_dim)
    b_o = np.zeros(vocab_size)
    return ArrayDict(w_e=w_e, w_h=w_h, w_o=w_o, b_h=b_h, b_o=b_o)


def random_params(init_wt: float, vocab_size: int, context_len: int,
                  embedding_dim: int, hidden_dim: int) -> ArrayDict:
    """
    Constructs an ArrayDict of randomly initalized parameters. Weights
    are initialized according to a normal distribution, and biases are
    initialized to 0.

    :param init_wt: The mean of the weights
    :param vocab_size: The number of possible words in the vocabulary
    :param context_len: The number of input units
    :param embedding_dim: The size of the word embeddings
    :param hidden_dim: The number of hidden units
    :return: The initialized parameters
    """
    w_e = np.random.normal(0., init_wt, size=(vocab_size, embedding_dim))
    w_h = np.random.normal(0., init_wt, size=(hidden_dim,
                                              context_len * embedding_dim))
    w_o = np.random.normal(0., init_wt, size=(vocab_size, hidden_dim))
    b_h = np.zeros(hidden_dim)
    b_o = np.zeros(vocab_size)
    return ArrayDict(w_e=w_e, w_h=w_h, w_o=w_o, b_h=b_h, b_o=b_o)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Task 3: Please implement sigmoid.

    :param z: A matrix
    :return: sigmoid(z)
    """
    f = lambda x : 1 / (1 + np.exp(-x))
    f = np.vectorize(f, otypes=[np.float])
    return f(z)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Task 3: Please implement softmax.

    :param z: A matrix of row vectors
    :return: A matrix where each row is the softmax of the corresponding
        row of z
    """
    def f(z: np.ndarray) -> np.ndarray:
        z -= np.max(z)
        g = lambda x : np.exp(x) / np.sum(np.exp(z))
        g = np.vectorize(g, otypes=[np.float])
        return g(z)
    
    return np.apply_along_axis(f, axis=1, arr=z)

class Model(object):
    """
    The neural network language model.
    """

    def __init__(self, params: ArrayDict, vocab: List[str]):
        """
        Initializes a Model from parameters and a vocabulary.

        :param params: The initial parameters of the model
        :param vocab: The vocabulary of the model
        """
        self.params = params
        self.vocab = vocab

        # Information about parameters
        self.vocab_size = len(vocab)
        self.embedding_dim = self.params["w_e"].shape[1]
        self.hidden_dim, self.embedding_layer_dim = self.params["w_h"].shape
        self.context_len = self.embedding_layer_dim // self.embedding_dim

        # Keep state
        self.input = None
        self.state = None
        self.grads = None
        self.delta_o = None

    @staticmethod
    def random_init(init_wt: float, vocab: List[str], context_len: int,
                    embedding_dim: int, hidden_dim: int):
        """
        Constructs a network with randomly initialized parameters.

        :param init_wt: The mean of the model weights
        :param vocab: The model vocabulary
        :param context_len: The number of input units
        :param embedding_dim: The size of the word embeddings
        :param hidden_dim: The number of hidden units
        :return: The randomly initialized network
        """
        params = random_params(init_wt, len(vocab), context_len,
                               embedding_dim, hidden_dim)
        return Model(params, vocab)

    @staticmethod
    def from_file(filename: str = "partially_trained.pk"):
        """
        Loads a model from a file.

        :param filename: The filename of the saved model
        :return: The loaded model
        """
        with open(filename, "rb") as f:
            model_params = pickle.load(f)

        vocab = model_params["vocab"]
        w_e = model_params["word_embedding_weights"]
        w_h = model_params["embed_to_hid_weights"]
        b_h = model_params["hid_bias"]
        w_o = model_params["hid_to_output_weights"]
        b_o = model_params["output_bias"]

        params = ArrayDict(w_e=w_e, w_h=w_h, w_o=w_o, b_h=b_h, b_o=b_o)
        return Model(params, vocab)

    def save(self, filename: str = "partially_trained.pk"):
        """
        Saves this model to a file.

        :param filename: The filename of the saved model
        :return: None
        """
        model_params = {"vocab": self.vocab,
                        "word_embedding_weights": self.params["w_e"],
                        "embed_to_hid_weights": self.params["w_h"],
                        "hid_bias": self.params["b_h"],
                        "hid_to_output_weights": self.params["w_o"],
                        "output_bias": self.params["b_o"]}
        with open(filename, "wb") as f:
            pickle.dump(model_params, f)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Implements the forward pass. This function computes all the
        activations and saves them to self.state.

        :param inputs: The model inputs (batch_size by D)
        :return: The model output
        """
        self.input = inputs
        batch_size = len(inputs)

        # Get embeddings
        embeddings = [self.params["w_e"][inputs[:, w]] for w in
                      range(self.context_len)]
        z_e = np.concatenate(embeddings, axis=1)

        # Task 4: Please implement the computation of z_h, a_h, z_o, and
        # a_o. Replace the placeholder code below with arrays of the 
        # same shape.
        
        # z_h = np.zeros((batch_size, self.hidden_dim))
        # a_h = np.zeros((batch_size, self.hidden_dim))
        # z_o = np.zeros((batch_size, self.vocab_size))
        # a_o = np.zeros((batch_size, self.vocab_size))

        w_h = self.params["w_h"]
        b_h = self.params["b_h"]

        f = lambda x : np.dot(w_h, x) + b_h
        z_h = np.apply_along_axis(f, axis=1, arr=z_e)

        a_h = sigmoid(z_h)

        w_o = self.params["w_o"]
        b_o = self.params["b_o"]

        g = lambda x : np.dot(w_o, x) + b_o
        z_o = np.apply_along_axis(g, axis=1, arr=a_h)

        a_o = softmax(z_o)

        # Save activations
        self.state = ArrayDict(z_e=z_e, z_h=z_h, z_o=z_o,
                               a_e=z_e, a_h=a_h, a_o=a_o)
        return a_o
    
    @staticmethod
    def reshape_tile(arr):
        return np.reshape(arr, (-1,1,arr.shape[-1]))

    def backward(self, loss_gradient: np.ndarray):
        """
        Implements the backward pass. This function computes all the
        gradients and saves them to self.grads.

        :return: None
        """

        delta_o = loss_gradient
        delta_h = (delta_o @ self.params["w_o"]) * \
                  self.state["a_h"] * (1. - self.state["a_h"])
        delta_e = delta_h @ self.params["w_h"]

        # Task 6: Compute the gradient of the loss with respect to each
        # of the weights and biases. Part of this has already been 
        # completed (the gradient with respect to w_e), but you need to 
        # fill in the derivative computations for w_o, b_o, w_h, and 
        # b_h.

        grad_w_h = np.zeros(self.params["w_h"].shape)
        grad_w_o = np.zeros(self.params["w_o"].shape)
        grad_b_h = np.zeros(self.params["b_h"].shape)
        grad_b_o = np.zeros(self.params["b_o"].shape)


#         for v in range(len(self.input)):

#             grad_w_h += (np.tile(self.state["a_e"][v], (self.params["w_h"].shape[0], 1)) * \
#                 np.tile(delta_h[v,:], (self.params["w_h"].shape[1], 1)).T)

#             grad_w_o += (np.tile(self.state["a_h"][v], (self.params["w_o"].shape[0], 1)) * \
#                 np.tile(delta_o[v,:], (self.params["w_o"].shape[1], 1)).T)

#             grad_b_h += delta_h[v,:]

#             grad_b_o += delta_o[v,:]
#         print('243', self.reshape_tile(self.state["a_e"]).shape)
#         print('244',(1,self.params["w_h"].shape[0], 1))
#         print(self.reshape_tile(delta_h).shape)
#         print((1, self.params["w_h"].shape[1], 1))
        
        
        grad_w_h = (np.tile(self.reshape_tile(self.state["a_e"]), (1,self.params["w_h"].shape[0], 1)) * \
            np.tile(self.reshape_tile(delta_h), (1, self.params["w_h"].shape[1], 1)).transpose((0,2,1))).sum(axis=0)
#         print(grad_w_h.shape)
        grad_w_o = (np.tile(self.reshape_tile(self.state["a_h"]), (1,self.params["w_o"].shape[0], 1)) * \
            np.tile(self.reshape_tile(delta_o), (1,self.params["w_o"].shape[1], 1)).transpose((0,2,1))).sum(axis=0)
#         print(grad_w_o.shape)
        grad_b_h = delta_h.sum(axis=0)
#         print(grad_b_h.shape)
        grad_b_o = delta_o.sum(axis=0)
#         print(grad_b_o.shape)

        # raise NotImplementedError("Please complete backward!")

        grad_w_e = np.zeros(self.params["w_e"].shape)
        delta_e_parts = np.split(delta_e, self.context_len, axis=1)

        for w in range(self.context_len):
            for b in range(len(self.input)):
                v = self.input[b, w]
                grad_w_e[v] += delta_e_parts[w][b]
        # Save the gradients
        self.grads = ArrayDict(w_e=grad_w_e, w_h=grad_w_h, w_o=grad_w_o,
                               b_h=grad_b_h, b_o=grad_b_o)