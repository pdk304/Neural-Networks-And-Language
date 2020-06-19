import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from cross_entropy import loss
from model import Model


def evaluate(model: Model, dataset_x: np.ndarray, dataset_y: np.ndarray,
             batch_size: int = 100) -> float:
    """
    Computes the average loss over a dataset for a given model.

    :param model: A model
    :param dataset_x: A dataset of inputs
    :param dataset_y: The correct outputs for the dataset
    :param batch_size: The size of each mini-batch
    :return: The average loss for this model over this dataset
    """
    total_loss = 0.
    for x, y in get_batches(dataset_x, dataset_y, batch_size):
        y_hat = model.forward(x)
        total_loss += loss(y_hat, y)

    return float(total_loss / len(dataset_x))


def display_nearest_words(model: Model, word: str, k: int = 10):
    """
    Lists the k words nearest to a given word, along with their
    distances.

    :param model: A model
    :param word: A word
    :param k: The number of nearest words to list
    :return: None
    """
    if word not in model.vocab:
        raise ValueError("{} is not in the model vocabulary.".format(word))

    # Compute distance to every other word
    idx = model.vocab.index(word)
    word_vec = model.params["w_e"][idx]
    diff = model.params["w_e"] - word_vec.reshape((1, -1))
    distance = np.linalg.norm(diff, axis=1)

    # Sort by distance
    order = np.argsort(distance)
    order = order[1:1 + k]  # The nearest word is the word; skip that
    print("Nearest words to {}".format(word))
    for i in order:
        print("{}: {}".format(model.vocab[i], distance[i]))


def predict_next_word(model: Model, *words: str, k: int = 10):
    """
    Lists the top k predictions for the next word along with their
    probabilities.

    Example usage:
        predict_next_word(model, "john", "might", "be", k=3)

    :param model: A model
    :param words: The input words for the model
    :param k: The number of predictions to list
    :return: None
    """
    if len(words) != model.context_len:
        msg = "The model needs {} input words.".format(model.context_len)
        raise ValueError(msg)
    for w in words:
        if w not in model.vocab:
            raise ValueError("{} is not in the model vocabulary.".format(w))

    idxs = [model.vocab.index(w) for w in words]
    x = np.array(idxs).reshape((1, -1))
    probs = model.forward(x).ravel()

    sorted_idxs = np.argsort(probs)[::-1]
    print(" ".join(words) + "...")
    for i in sorted_idxs[:k]:
        print("{}: {}".format(model.vocab[i], probs[i]))


def l2_dist(model: Model, word1: str, word2: str) -> np.float:
    """
    Computes the Eulidean (L2) distance between two word vectors.

    :param model: A model
    :param word1: A word
    :param word2: Another word
    :return: The Euclidean distance between the embeddings for word1 and
        word2 according to model.w_e
    """
    if word1 not in model.vocab:
        raise ValueError("{} is not in the model vocabulary.".format(word1))
    if word2 not in model.vocab:
        raise ValueError("{} is not in the model vocabulary.".format(word2))

    word1_vec = model.params["w_e"][model.vocab.index(word1)]
    word2_vec = model.params["w_e"][model.vocab.index(word2)]
    return np.linalg.norm(word1_vec - word2_vec)


def cos_sim(model: Model, word1: str, word2: str) -> np.float:
    """
    Computes the cosine similarity between two word vectors.

    :param model: A model
    :param word1: A word
    :param word2: Another word
    :return: The cosine similarity between the embeddings for word1 and
        word2 according to model.w_e
    """
    if word1 not in model.vocab:
        raise ValueError("{} is not in the model vocabulary.".format(word1))
    if word2 not in model.vocab:
        raise ValueError("{} is not in the model vocabulary.".format(word2))

    word1_vec = model.params["w_e"][model.vocab.index(word1)]
    word2_vec = model.params["w_e"][model.vocab.index(word2)]
    return np.dot(word1_vec, word2_vec) / \
           (np.linalg.norm(word1_vec) * np.linalg.norm(word2_vec))


def plot_word_vectors(model: Model, init_dim: int = 50):
    """
    Plots the word vectors of a model using t-SNE.

    :param model: A model
    :param init_dim: The t-SNE visualization is initialized by using PCA
        to reduce the embedding space to this many dimensions. The value
        of init_dim must be smaller than the embedding size
    :return: None
    """
    if init_dim > model.embedding_dim:
        raise ValueError("init_dim needs to be smaller than the embedding "
                         "size of {}!".format(model.embedding_dim))

    pca = PCA(n_components=init_dim)
    embeddings_pca = pca.fit_transform(model.params["w_e"])

    tsne = TSNE(n_components=2, learning_rate=500., init="pca")
    embeddings_tsne = tsne.fit_transform(embeddings_pca)

    # Plot
    plt.figure()
    for i, w in enumerate(model.vocab):
        plt.text(embeddings_tsne[i, 0], embeddings_tsne[i, 1], w)
    plt.xlim(embeddings_tsne[:, 0].min(), embeddings_tsne[:, 0].max())
    plt.ylim(embeddings_tsne[:, 1].min(), embeddings_tsne[:, 1].max())

    plt.show()


def get_batches(xs: np.ndarray, ys: np.ndarray, batch_size: int,
                shuffle: bool = True):
    """
    Divides a dataset into mini-batches of a given size.

    :param xs: The inputs (num_examples by context_len)
    :param ys: The outputs (num_examples)
    :param batch_size: The size of each mini-batch
    :param shuffle: If true, the mini-batches will be shuffled
    :return: None (this is a generator)
    """
    num_examples = len(xs)
    if num_examples % batch_size != 0:
        raise RuntimeError("Cannot divide {} examples into mini-batches of "
                           "size {}.".format(num_examples, batch_size))
    num_batches = num_examples // batch_size

    if shuffle:
        idxs = np.random.permutation(num_examples)
        xs = xs[idxs, :]
        ys = ys[idxs]

    for m in range(num_batches):
        i_start = m * batch_size
        i_end = (m + 1) * batch_size
        yield xs[i_start:i_end], ys[i_start:i_end]
