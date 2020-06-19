import torch
import torch.nn as nn
import torchtext.data as tt

class FFPOSTagger(nn.Module):
    """
    The feedforward POS tagging network.
    """

    def __init__(self, text_field: tt.Field, tag_field: tt.Field,
                 context_size: int, embedding_dim: int, hidden_dim: int):
        """
        Constructs a FFPOSTagger.

        :param text_field: The Field for model inputs
        :param tag_field: The Field for model outputs (POS tags)
        :param context_size: The context size used by the model
        :param embedding_dim: The size of the word embeddings
        :param hidden_dim: The size of the hidden layer output
        """
        super(FFPOSTagger, self).__init__()

        self.text_field = text_field
        self.tag_field = tag_field
        self.context_size = context_size
        self.window_size = 2 * context_size + 1
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Problem 2a: Insert your code here. You will need to define the
        # embedding and linear layers for your feedforward network.
        self._embedding = nn.Embedding(len(text_field.vocab), self.embedding_dim)
        self._linear1 = nn.Linear(self.window_size * self.embedding_dim, self.hidden_dim)
        self._linear2 = nn.Linear(self.hidden_dim, len(tag_field.vocab))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass here. First, populate the
        input_embedding variable, which should contain a tensor of shape
        (batch_size, window_size * embedding_dim). Then, pass this
        tensor through the network and return the (pre-softmax) result,
        which should be of shape (batch_size, len(tag_field.vocab)).

        :param x: The model input, represented as a tensor of shape
            (window_size, batch_size) containing word indices
        :return: The model output, whose shape is (batch_size, len(tag_field.vocab))
        """
        # Replace the following line with your own code and fill in below

        input_embedding = self._embedding(x).permute(1,0,2).reshape(x.shape[1], self.window_size*self.embedding_dim)
        hidden = torch.sigmoid(self._linear1(input_embedding))
        return self._linear2(hidden)
