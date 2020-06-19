import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.data as tt
from sklearn.metrics import confusion_matrix

from q3_model import RNNPOSTagger
from tools import print_cm, load_ud_pos_data, EarlyStopping


def train(model: RNNPOSTagger, train_iter: tt.Iterator, val_iter: tt.Iterator,
          optimizer: optim.Optimizer, criterion: nn.Module, epochs: int,
          short_train: bool = False, patience: int = 3):
    """
    Trains a model.

    :param model: The model to train
    :param train_iter: The training iterator
    :param val_iter: The validation iterator
    :param optimizer: The optimizer
    :param criterion: The loss function
    :param epochs: The number of epochs to train for
    :param short_train: If True, only train every 20 batches
    :param patience: The max # of loss-increasing epochs before early stopping 
    :return: None
    """
    early_stopping = EarlyStopping(patience=patience, verbose=False, filename='q3_checkpoint.pt')
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0.
        total_batches = 0
        for batch_idx, batch in enumerate(train_iter):
            if short_train and batch_idx % 20 != 0:
                continue

            # Forward pass
            optimizer.zero_grad()
            output, _ = model.forward(batch.text)
            batch_loss = 0.
            for i in range(output.size()[0]):
                batch_loss += criterion(output[i, :, :].squeeze(),
                                        batch.udtags[i, :].squeeze())

            # Backward pass
            batch_loss.backward(retain_graph=True)
            optimizer.step()

            # Record the total loss
            epoch_loss += batch_loss
            total_batches += 1

        # Finish training, print status to stdout
        duration = (time.time() - start_time)
        accuracy, unk_accuracy, val_loss = evaluate(model, val_iter, criterion)
        print("Epoch %d" % epoch, end=": ")
        print("loss per batch = %.4f" % (epoch_loss / total_batches), end=", ")
        print("val loss = %.4f" % val_loss, end=", ")
        print("val acc = %.4f" % accuracy, end=", ")
        print("unk acc = %.4f" % unk_accuracy, end=" ")
        print("(%.3f sec)" % duration)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping, reloading checkpoint model")
            model.load_state_dict(torch.load('q3_checkpoint.pt'))
            break

def eval_sent(model: RNNPOSTagger, sentence_list: List[str]) -> \
        List[List[str]]:
    """
    Runs the model on a list of sentences.

    :param model: The model to run
    :param sentence_list: A list of input sentences that the model will
        tag. Each sentence is a string of words separated by spaces. See
        the script at the bottom for an example
    :return: The POS tags of each sentence in sentence_list
    """
    predictions = []
    sentences = [model.text_field.tokenize(s) for s in sentence_list]
    indices = model.text_field.process(sentences)
    model_output = model.forward(indices)[0].argmax(2)
    for i in range(len(sentences)):
        predictions.append([model.tag_field.vocab.itos[j]
                            for j in model_output[1:-1, i]])

    return predictions


def evaluate(model: RNNPOSTagger, eval_iter: tt.Iterator, criterion: nn.Module,
             conf_matrix: bool = False) -> Tuple[float, float]:
    """
    Evaluates a model.

    :param model: The model to evaluate
    :param eval_iter: The testing or validation iterator
    :param conf_matrix: If true, a confusion matrix will be printed to
        stdout
    :return: The model's accuracy on the provided dataset
    """
    pad_token = model.tag_field.pad_token
    bos_token = model.tag_field.init_token
    eos_token = model.tag_field.eos_token
    unk_token = model.text_field.unk_token

    pad_index = model.tag_field.vocab.stoi[pad_token]
    bos_index = model.tag_field.vocab.stoi[bos_token]
    eos_index = model.tag_field.vocab.stoi[eos_token]
    unk_index = model.text_field.vocab.stoi[unk_token]

    pads, correct, tries, unks, unk_correct, loss = 0., 0., 0., 0., 0., 0.
    pred_list, target_list = np.array([]), np.array([])
    for i, batch in enumerate(eval_iter):
        sentence_length, batch_size = batch.text.size()

        # Mask out <pad>, <bos>, and <eos>
        pad_mask = (batch.udtags == pad_index)
        bos_mask = (batch.udtags == bos_index)
        eos_mask = (batch.udtags == eos_index)
        other_mask = pad_mask | bos_mask | eos_mask
        others = torch.sum(other_mask)

        # Count the number of <unks>
        unk_mask = (batch.text == unk_index)
        unks += torch.sum(unk_mask)

        # Count the total number of words evaluated
        tries += (sentence_length * batch_size) - others

        # Compute model output
        
        y_hat, _ = model.forward(batch.text)
        model_output = y_hat.argmax(2)

        #compute batch loss 
        for i in range(sentence_length):
            loss += criterion(y_hat[i,:,:].squeeze(), batch.udtags[i,:].squeeze())

        target_list = np.concatenate((target_list,
                                      batch.udtags.view(-1).numpy()))
        pred_list = np.concatenate((pred_list, model_output.view(-1).numpy()))

        # Count correct predictions
        correct_mask = (batch.udtags == model_output) & ~other_mask
        correct += torch.sum(correct_mask)
        unk_correct += torch.sum(unk_mask & correct_mask)

    # Compute accuracy
    accuracy = correct / tries
    unk_accuracy = unk_correct / unks

    # Display a confusion matrix
    if conf_matrix:
        target_list = [model.tag_field.vocab.itos[int(target)]
                       for target in target_list]
        pred_list = [model.tag_field.vocab.itos[int(pred)]
                     for pred in pred_list]
        cm = confusion_matrix(target_list, pred_list,
                              labels=model.tag_field.vocab.itos)
        print_cm(cm, model.tag_field.vocab.itos)

    return accuracy.item(), unk_accuracy.item(), (loss.item() / i)


if __name__ == "__main__":
    # Problem 3d: Use this script to test your code.
    # My results for the hyperparameter tuning are in a separate text file.
    train_iter, val_iter, test_iter, text_field, tag_field = \
        load_ud_pos_data(10, min_freq=2)

    # Set up the model
    tagger = RNNPOSTagger(text_field, tag_field, 20, 10, bidir=False)
    pad_token = tag_field.pad_token
    pad_index = tag_field.vocab.stoi[pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = optim.Adam(tagger.parameters(), lr=1e-3)

    # Train the model
    train(tagger, train_iter, val_iter, optimizer, criterion, 40,
          short_train=True)

    # Test the model
    sent = ["if you push it , it will never work ."]
    print(sent[0], "POS tags:", ' '.join(eval_sent(tagger, sent)[0]))
    print("test set acc = %.4f, test set unk acc = %.4f, test set loss = %.4f" %
        evaluate(tagger, test_iter, criterion))
