import pickle

import numpy as np

from cross_entropy import loss, loss_grad
from evaluate import evaluate, get_batches, predict_next_word, \
    plot_word_vectors
from model import Model, zero_params


def train(embedding_dim, hidden_dim, data_file: str = "data.pk",
          batch_size: int = 100, learning_rate: float = .2, momentum: float
          = .9, epochs: int = 50, init_wt: float = .01, context_len: int = 3,
          show_training_loss_after: int = 100,
          show_validation_loss_after: int = 1000) -> Model:
    """
    This is the main training routine for the language model.

    :param embedding_dim: The size of the word embeddings

    :param hidden_dim: The number of units in the hidden layer

    :param data_file: The .pk file containing the training, testing, and
        validation data

    :param batch_size: The size of each mini-batch

    :param learning_rate: The learning rate used for stochastic gradient
        descent

    :param momentum: The momentum for stochastic gradient descent

    :param epochs: The maximum number of epochs to train for

    :param init_wt: The standard deviation of the initial random weights

    :param context_len: The number of input words to the model

    :param show_training_loss_after: Measure the training loss after
        this many mini-batches

    :param show_validation_loss_after: Measure the validation loss after
        this many mini-batches

    :return: The trained model
    """
    # Load and unpack the training, validation and test data
    with open(data_file, "rb") as f:
        data_obj = pickle.load(f)

    vocab = data_obj["vocab"]
    train_xs, train_ys = data_obj["train_inputs"], data_obj["train_targets"]
    valid_xs, valid_ys = data_obj["valid_inputs"], data_obj["valid_targets"]
    test_xs, test_ys = data_obj["test_inputs"], data_obj["test_targets"]

    # Create the model and randomly initialize the trainable parameters
    model = Model.random_init(init_wt, vocab, context_len, embedding_dim,
                              hidden_dim)

    # Initialize the momentum vector to all zeros
    delta = zero_params(len(vocab), context_len, embedding_dim,
                        hidden_dim)

    total_loss = 0.
    batch_count = 0
    best_valid_loss = np.infty
    end_training = False
    for epoch in range(1, epochs + 1):
        print("Epoch", epoch)

        batches = get_batches(train_xs, train_ys, batch_size)
        for m, (input_batch, target_batch) in enumerate(batches):
            batch_count += 1

            # Forward pass
            y_hat = model.forward(input_batch)

            # Measure loss function
            batch_loss = loss(y_hat, target_batch) / batch_size
            total_loss += batch_loss
            if batch_count % show_training_loss_after == 0:
                avg_loss = total_loss / show_training_loss_after
                print("Batch {} Loss: {:1.3f}".format(batch_count, avg_loss))
                total_loss = 0.

            # Compute loss gradient with respect to network outputs
            delta_o = loss_grad(y_hat, target_batch)
            delta_o /= batch_size

            # Backward pass
            model.backward(delta_o)

            # Update the momentum vector and model parameters
            delta = momentum * delta + model.grads
            model.params -= learning_rate * delta

            # Early stopping
            if batch_count % show_validation_loss_after == 0:
                print("Running validation...")
                valid_loss = evaluate(model, valid_xs, valid_ys)
                print("Validation loss: {:1.3f}".format(valid_loss))

                if valid_loss > best_valid_loss:
                    print("Validation error increasing! Training stopped.")
                    end_training = True
                    break

                best_valid_loss = valid_loss

        if end_training:
            break

    test_loss = evaluate(model, test_xs, test_ys)
    print("Final testing loss: {:1.3f}".format(test_loss))

    return model


if __name__ == "__main__":
    model = train(25, 100)
    predict_next_word(model, "the", "man", "is")
    plot_word_vectors(model, init_dim=5)
