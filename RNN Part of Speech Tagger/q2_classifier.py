import time
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext.data as tt
from sklearn.metrics import confusion_matrix

from q2_model import FFPOSTagger
from tools import print_cm, load_ud_pos_data, EarlyStopping


def add_padding(text: torch.Tensor, model: FFPOSTagger) -> torch.Tensor:
    """
    Pads a sequence of words, represented as a tensor of indices, to a
    minimum window size.

    :param text: The word sequence to be padded
    :param model: The model that determines the window size
    :return: The padded word sequence
    """
    pad_token = model.text_field.pad_token
    pad_index = model.text_field.vocab.stoi[pad_token]
    
    # Problem 2b: Fill in your code here.
    k = model.context_size
    return F.pad(text.T, pad=(k,k), mode='constant', value=pad_index).T


def train(model: FFPOSTagger, train_iter: tt.Iterator, val_iter: tt.Iterator,
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
    early_stopping = EarlyStopping(patience=patience, verbose=False, filename='q2_checkpoint.pt')
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss, total_batches = 0., 0.
        for batch_idx, batch in enumerate(train_iter):
            if short_train and batch_idx % 20 != 0:
                continue
            total_batches += 1

            text = add_padding(batch.text, model)
            example_length, batch_size = batch.text.shape
            for i in range(example_length):
                
                # forward pass 
                x = text[i:i+model.window_size]
                y_hat = model(x)
                y = batch.udtags[i]
                loss = criterion(y_hat,y)
                
                # backward pass
                model.zero_grad()
                loss.backward()
                optimizer.step()
                
                #keep track of progress
                epoch_loss += loss


        # Print status to stdout
        duration = (time.time() - start_time)
        valid_accuracy, valid_loss = evaluate(model, val_iter, criterion)
        print("Epoch %d:" % epoch, end=" ")
        print("loss per batch = %.2f" % (epoch_loss / total_batches), end=", ")
        print("val loss = %.4f, val acc = %.4f" % (valid_loss, valid_accuracy), end=" ")
        print("(%.3f sec)" % duration)
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping, reloading checkpoint model")
            model.load_state_dict(torch.load('q2_checkpoint.pt'))
            break


def evaluate(model: FFPOSTagger, eval_iter: tt.Iterator, criterion: nn.Module,
             conf_matrix: bool = False) -> Union[float, torch.Tensor]:
    """
    Evaluates a model.

    :param model: The model to evaluate
    :param eval_iter: The testing or validation iterator
    :param criterion: The loss function
    :param conf_matrix: If true, a confusion matrix will be printed to
        stdout
    :return: The model's accuracy on the provided dataset
    """
    accuracy, loss = 0., 0.
    pred_list = np.array([])  # Needed for confusion matrix, see below
    target_list = np.array([])  # Needed for confusion matrix, see below
    
    pad_token = model.tag_field.pad_token
    bos_token = model.tag_field.init_token
    eos_token = model.tag_field.eos_token
    
    pad_index = model.tag_field.vocab.stoi[pad_token]
    bos_index = model.tag_field.vocab.stoi[bos_token]
    eos_index = model.tag_field.vocab.stoi[eos_token]
    
    
    total_batches = 0

    correct = 0
    tries = 0
    
    for batch in eval_iter:
        
        # Mak out <pad>, <bos>, <eos>
        pad_mask = (batch.udtags == pad_index)
        bos_mask = (batch.udtags == bos_index)
        eos_mask = (batch.udtags == eos_index)
        other_mask = pad_mask | bos_mask | eos_mask
        others = torch.sum(other_mask)
        
        text = add_padding(batch.text, model)
        example_length, batch_size = batch.text.shape

        # Problem 2d: Fill in your code here.
        tries += (example_length * batch_size) - others
        
        for i in range(example_length):   
            
            # Compute model ouput
            x = text[i:i+model.window_size]
            y_hat = model(x)
            model_output = y_hat.argmax(axis=1)
            
            # Compute batch loss
            y = batch.udtags[i]
            loss += criterion(y_hat,y)
            
            target_list = np.concatenate((target_list, y.cpu().numpy()))
            pred_list = np.concatenate((pred_list, model_output.cpu().numpy()))
            
            # Count correct predictions
            pad_mask = (y == pad_index)
            bos_mask = (y == bos_index)
            eos_mask = (y == eos_index)
            other_mask = pad_mask | bos_mask | eos_mask
            
            correct_mask = (y == model_output) & ~other_mask
            correct += torch.sum(correct_mask)
            
        total_batches += 1
    
    loss = float(loss) / float(total_batches)
    accuracy = float(correct) / float(tries)

    # If you keep a list of the network outputs and the targets in the
    # variables pred_list and target_list, the following lines of code
    # will create and print a confusion matrix for you.
    if conf_matrix:
        target_list = [model.tag_field.vocab.itos[int(target)]
                       for target in target_list]
        pred_list = [model.tag_field.vocab.itos[int(pred)]
                     for pred in pred_list]
        cm = confusion_matrix(target_list, pred_list,
                              labels=model.tag_field.vocab.itos)
        print_cm(cm, model.tag_field.vocab.itos)

    return accuracy, loss


def eval_sents(model: FFPOSTagger, sentence_list: List[str]) \
        -> List[List[str]]:
    """
    Runs the model on a list of sentences.

    :param model: The model to run
    :param sentence_list: A list of input sentences that the model will
        tag. Each sentence is a string of words separated by spaces. See
        the script at the bottom for an example
    :return: The POS tags of each sentence in sentence_list
    """
    model.cuda()
    predictions = []
    sentences = [model.text_field.tokenize(s) for s in sentence_list]
    indices = model.text_field.process(sentences)
    input_ = add_padding(indices, model)

    # Problem 2e: Fill in your code here.
    for sentence in sentences:
        for i in range(len(indices)):   
            x = input_[i:i+model.window_size]
            y_hat = model(x)
            guess = y_hat.argmax(axis=1)
            predictions.append(model.tag_field.vocab.itos[guess])

    return predictions

if __name__ == "__main__":
    # Use this script to test your code.
    # My results for the hyperparameter tuning question is in a separate  text file.
    train_iter, val_iter, test_iter, text_field, tag_field = \
        load_ud_pos_data(10, min_freq=2)
    
    # Set up model (default paremeters)
    tagger = FFPOSTagger(text_field, tag_field, 2, 20, 10)
    pad_token = tagger.text_field.pad_token
    pad_index = tagger.text_field.vocab.stoi[pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = optim.Adam(tagger.parameters(), lr=1e-2)
    
    # Train
    train(tagger, train_iter, val_iter, optimizer, criterion, 50,
          short_train=True, patience=5)
    
    # Sanity check
    test_sentences = ["if you push it , it will never work ."]
    predicted_pos = eval_sents(tagger, test_sentences)
    print(test_sentences[0], "Predicted POS tags:", predicted_pos)
    
    test_accuracy, _ = evaluate(tagger, test_iter, criterion)
    print("Test set accuracy = %.4f" % test_accuracy)