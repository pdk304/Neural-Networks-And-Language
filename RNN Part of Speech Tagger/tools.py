from typing import List, Tuple

import numpy as np
import torch
import torchtext.data as tt
from torchtext.datasets import UDPOS


def print_cm(cm: np.ndarray, labels: List[str], hide_zeroes: bool = False,
             hide_diagonal: bool = False, hide_threshold: float = None):
    """
    Prints a confusion matrix to stdout.

    :param cm: A confusion matrix
    :param labels: The labels of the confusion matrix
    :param hide_zeroes: If True, 0s will not be shown
    :param hide_diagonal: If True, the diagonal will not be shown
    :param hide_threshold: Values below this number will not be shown
    :return: None
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()

    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold is not None:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def load_ud_pos_data(batch_size: int, min_freq: int = 10) -> \
        Tuple[tt.Iterator, tt.Iterator, tt.Iterator, tt.Field, tt.Field]:
    """
    Loads the Universal Dependencies POS tagging dataset. The first time you call this function, it will
    download the data and save it to a local directory. The data will
    be loaded from your local disk if it is available.

    We will add special beginning-of-sentence and end-of-sentence tokens
    (<bos> and <eos>, respectively) to our vocabulary and to our set of
    POS tags.

    :param batch_size: The size of the mini-batches
    :param min_freq: A word will only be added to the vocabulary if it
        occurs this many times in the data
    :return: Iterators for the three datasets, along with the Fields
        for words and POS tags
    """
    # Prepare fields
    text_field = tt.Field(init_token="<bos>", eos_token="<eos>", lower=True)
    tags_field = tt.Field(init_token="<bos>", eos_token="<eos>",
                          unk_token=None)
    fields = (("text", text_field), ("udtags", tags_field))

    # Load data
    train_data, valid_data, test_data = UDPOS.splits(fields)

    # Prepare vocab
    text_field.build_vocab(train_data, valid_data, test_data,
                           min_freq=min_freq)
    tags_field.build_vocab(train_data, valid_data, test_data)
    print(f"Size of TEXT vocabulary: {len(text_field.vocab)}")
    print(f"Size of UD_TAG vocabulary: {len(tags_field.vocab)}")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #  device = torch.device('cuda')
    device = torch.device('cpu')
    iters = tt.BucketIterator.splits((train_data, valid_data, test_data),
                                     batch_size=batch_size, device=device)

    return iters + (text_field, tags_field)

class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, filename='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.filename = filename

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.filename)
        self.val_loss_min = val_loss