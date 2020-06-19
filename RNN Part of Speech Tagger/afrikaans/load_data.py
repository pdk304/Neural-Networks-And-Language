from typing import Tuple

import torch
from torchtext import data as tt
from torchtext.datasets import SequenceTaggingDataset


def load_af(batch_size: int, min_freq: int = 10, nl_ratio: float = .5) -> \
        Tuple[tt.Iterator, tt.Iterator, tt.Dataset, tt.Field, tt.Field]:
    """
    Loads the Afrikaans data, augmented with Dutch data.

    :param batch_size: The size of the mini-batches
    :param min_freq: A word will only be added to the vocabulary if it
        occurs this many times in the data
    :param nl_ratio: The fraction of the training data that will be
        Dutch
    :return: Iterators for the three datasets, along with the Fields
        for words and POS tags. Only the training data will contain
        Dutch examples
    """
    if not 0 <= nl_ratio <= 1:
        raise ValueError("nl_ratio must be between 0 and 1")

    # Prepare fields
    text_field = tt.Field(init_token="<bos>", eos_token="<eos>", lower=True)
    tags_field = tt.Field(init_token="<bos>", eos_token="<eos>",
                          unk_token=None)
    fields = (("text", text_field), ("udtags", tags_field))

    # Load data
    af = list(SequenceTaggingDataset.splits(path="data/af", fields=fields,
                                            train="train.txt", test="test.txt",
                                            validation="dev.txt"))
    nl_train = SequenceTaggingDataset("data/nl/nl.txt", fields)

    # Add Dutch examples
    max_nl_ratio = len(nl_train) / (len(af[0]) + len(nl_train))
    if nl_ratio <= max_nl_ratio:
        num_nl_examples = int(nl_ratio * len(af[0]) / (1. - nl_ratio))
        af[0].examples += nl_train.examples[:num_nl_examples]
    else:
        num_af_examples = int(len(nl_train) * (1. - nl_ratio) / nl_ratio)
        af[0].examples = af[0].examples[:num_af_examples] + nl_train.examples

    # Build vocab
    text_field.build_vocab(*af, min_freq=min_freq)
    tags_field.build_vocab(*af)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    iters = tt.BucketIterator.splits(af, batch_size=batch_size, device=device)

    return iters + (text_field, tags_field)


if __name__ == "__main__":
    train_iter, dev_iter, test_iter, text_field, tags_field = load_af(32)
    for batch in train_iter:
        print(batch)
        break
