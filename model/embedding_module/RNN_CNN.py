import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class RNN_CNN(nn.Module):
    # embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu
    def __init__(self, embedding_dim=100, hidden_dim=64):

        super(RNN_CNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = 64
        self.use_gpu = torch.cuda.is_available()
        self.batch_size = 52

        self.content_dim = hidden_dim
        vocab_size = 24

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings.weight = nn.Parameter(self.embeddings)
        # self.word_embeddings.weight.data.copy_(torch.from_numpy(self.embeddings))
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)
        # self.hidden2label = nn.Linear(self.hidden_dim, self.label_size)
        self.hidden = self.init_hidden()
        window_size = 3
        self.conv = nn.Conv1d(in_channels=hidden_dim, out_channels=self.content_dim, kernel_size=window_size,
                              padding = (window_size - 1) // 2)
        self.classification = nn.Linear(self.content_dim, 2)
        # self.properties.update({"content_dim": self.content_dim})

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.use_gpu:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, x):
        # print(x.shape)
        # print(x.shape[0])

        x = x.cuda() # [4019, 52]
        embeds = self.word_embeddings(x)  # [4019, 52, 100]

        x = embeds.view(x.shape[1], x.shape[0], -1) # [52, 4019, 100]
        # x = embeds.permute(1, 0, 2)
        hidden = self.init_hidden(x.shape[1])  # ([1, 4019, 64], [1, 4019, 64])
        r, (hidden, c) = self.lstm(x,hidden)
        hidden = hidden.permute(1, 2, 0)
        lstm_out = hidden # [4019, 64, 1]
        representation = self.conv(lstm_out)  # [4019, 64, 1]
        representation = representation.squeeze(2)
        return representation