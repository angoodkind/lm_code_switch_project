import torch.nn as nn
import torch
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, n_speakers, vocab_size, embedding_dim, hidden_dim, n_classes, n_layers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.word_encoder = nn.Embedding(vocab_size, embedding_dim)
        self.spkr_encoder = nn.Embedding(n_speakers, embedding_dim)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, nonlinearity=nonlinearity, dropout=dropout)

        # The linear layer that maps from hidden state space to the classes (code-switch / no code-switch)
        self.hidden2tag = nn.Linear(hidden_dim, n_classes)

        # store architecture hyperparameters
        self.rnn_type = rnn_type
        self.nhid = hidden_dim
        self.nlayers = n_layers

        # initialize weights
        self.hidden = self.init_hidden()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.word_encoder.weight.data.uniform_(-initrange, initrange)
        self.spkr_encoder.weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.bias.data.fill_(0)
        self.hidden2tag.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_data, hidden):
        # hidden is an external variable here, so it can be reset outside the model
        word_emb = self.drop(self.word_encoder(input_data[0]))
        spkr_emb = self.drop(self.spkr_encoder(input_data[1]))
        # add word and speaker information
        emb = word_emb + spkr_emb

        rnn_output, hidden = self.rnn(emb.view(emb.size(0), 1, -1), hidden)
        rnn_output = self.drop(rnn_output)
        unlogged_outputs = self.hidden2tag(rnn_output.view(emb.size(0), -1))
        return unlogged_outputs, hidden

    def init_hidden(self, bsz = 1):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
    # def init_hidden(self):
    #     # Before we've done anything, we dont have any hidden state.
    #     # Refer to the Pytorch documentation to see exactly
    #     # why they have this dimensionality.
    #     # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    #     return (Variable(torch.zeros(1, 1, self.nhid)),
    #             Variable(torch.zeros(1, 1, self.nhid)))
