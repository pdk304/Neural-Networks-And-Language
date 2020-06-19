import torch
import torch.nn as nn
import torchtext.data as tt


class RNNPOSCell(nn.Module):
    """
    A simple recurrent network unit. At each time step, the RNNPOSCell
    will take a word embedding and the previous hidden state as input
    and produce a new hidden state as output.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        Constructs an RNNPOSCell.

        :param embedding_dim: The size of word embeddings
        :param hidden_dim: The size of the hidden state
        """
        super(RNNPOSCell, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Create a (linear) layer of appropriate dimensions to be used 
        # in the cell's computations. 
        
        self.hidden = nn.Linear(hidden_dim + embedding_dim, hidden_dim)

    def forward(self, embedded_input: torch.Tensor,
                hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Problem 3b: Implement the RNNPOSCell forward pass here. This
        function should only implement the forward pass for one time
        step. It takes a single word embedding and hidden state as input
        and outputs a single hidden state.

        :param embedded_input: A word embedding, of shape (batch_size,
            embedding_dim)
        :param hidden_state: The previous hidden state, of shape
            (batch_size, hidden_dim)
        :return: The next hidden state, of shape (batch_size,
            hidden_dim)
        """
        concatenated = torch.cat((embedded_input, hidden_state), 1)
        hidden = torch.sigmoid(self.hidden(concatenated))

        return hidden

class RNNPOSTagger(nn.Module):
    """
    A POS tagger consisting of an embedding layer, hidden layer, and
    output layer. The hidden layer is a bidirectional simple recurrent
    network unit.
    """

    def __init__(self, text_field: tt.Field, tag_field: tt.Field,
                 embedding_dim: int, hidden_dim: int, bidir: bool=True):
        """
        Constructs an RNNPOSTagger.

        :param text_field: The Field for model inputs
        :param tag_field: The Field for model outputs (POS tags)
        :param embedding_dim: The size of the word embeddings
        :param hidden_dim: The size of the hidden layer output
        :param bidir: A boolean specifying if the tagger is bidirectional
        """
        super(RNNPOSTagger, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_field = tag_field
        self.text_field = text_field
        self.bidir = bidir

        self.embedding = nn.Embedding(len(text_field.vocab), embedding_dim)

        self.forward_cell = RNNPOSCell(self.embedding_dim, self.hidden_dim)
        if bidir:
            self.backward_cell = RNNPOSCell(self.embedding_dim, self.hidden_dim)

        self.output = nn.Linear(hidden_dim * (2 if bidir else 1), len(tag_field.vocab))

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of an RNNPOSTagger. This function
        should implement the forward pass for all time steps of the SRN
        units. It takes a sequence of word indices and outputs a
        sequence of POS tags.

        :param text: A sequence of words, of shape (sentence_length,
            batch_size)
        :return: A sequence of output probabilities, of shape
            (sentence_length, batch_size, len(self.tagset))
        """
        sentence_length, batch_size = text.shape 

        # Get embeddings, of shape (sentence_length, batch_size,
        # embedding_dim)
        input_embedding = self.embedding(text)

        #initialize hidden and output vectors of appropriate size
        lr_tensor = torch.zeros(sentence_length, batch_size, self.hidden_dim)
        if self.bidir:
            rl_tensor = torch.zeros(sentence_length, batch_size, self.hidden_dim)

        hidden_lr = torch.zeros(batch_size, self.hidden_dim)
        if self.bidir:
            hidden_rl = torch.zeros(batch_size, self.hidden_dim)

        hidden = torch.zeros(sentence_length, batch_size, self.hidden_dim*(2 if self.bidir else 1))

        out = torch.zeros(sentence_length, batch_size,
                          len(self.tag_field.vocab))

        # Problem 3c: Fill in your code here.
        for i in range(sentence_length):
            
            hidden_lr = self.forward_cell.forward(input_embedding[i], hidden_lr)
            lr_tensor[i] = hidden_lr

        if self.bidir:
            for i in range(sentence_length):
                
                hidden_rl = self.backward_cell.forward(input_embedding[sentence_length-i-1], hidden_rl)
                rl_tensor[i] = hidden_rl

                
        if self.bidir:
            hidden = torch.cat((lr_tensor, rl_tensor),2)
            for i in range(sentence_length):
                out = self.output(hidden)
        else:
            hidden = lr_tensor
            for i in range(sentence_length):
                out = self.output(hidden)

        return out, hidden
