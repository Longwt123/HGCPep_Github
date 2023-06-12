import torch
import numpy as np
import torch.nn as nn
from sklearn.utils import shuffle
from torch.autograd import Variable


class LSTMAttention(torch.nn.Module):
    def __init__(self, input_dim=100, hidden_units=64):

        super(LSTMAttention, self).__init__()
        # self.embedding_dim = 128
        # self.hidden_dim = 128
        self.embedding_dim = input_dim
        self.hidden_dim = hidden_units
        self.batch_size = 52
        self.use_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding(24, self.embedding_dim)
        # self.word_embeddings.weight = nn.Parameter(config.embeddings, requires_grad=config.embedding_training)

        self.num_layers = 1
        # self.bidirectional = True
        self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, batch_first=True, num_layers=self.num_layers,
                              dropout=0.2, bidirectional=True)
        self.hidden2label = nn.Linear(self.hidden_dim, 2)
        self.hidden = self.init_hidden()
        self.attn_fc = torch.nn.Linear(self.embedding_dim, 1)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.use_gpu:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).cuda())
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).cuda())
        else:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
        return (h0, c0)

    def attention(self, rnn_out, state):
        # print(state.shape)
        merged_state = torch.cat([s for s in state], 1)
        # print(merged_state.shape)
        merged_state = merged_state.unsqueeze(2)
        # print(merged_state.shape)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

    # end method attention

    def forward(self, x):
        x = x.cuda()
        embedded = self.word_embeddings(x) # [4019, 52, 100]
        hidden = self.init_hidden(x.size()[0])  # ([4, 4019, 32], [4, 4019, 32])
        rnn_out, hidden = self.bilstm(embedded, hidden) # [4019, 52, 64]
        h_n, c_n = hidden
        attn_out = self.attention(rnn_out, h_n) # [4019, 64]
        # logits = self.hidden2label(attn_out) # [4019, 2]
        return attn_out