import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size=24, emb_dim=100, hidden_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        layer_num = 2
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=layer_num)
        self.h0 = torch.randn(layer_num, 52, hidden_dim)  # 初始隐藏状态，层数为2，批次大小为3，隐藏维度为20
        self.c0 = torch.randn(layer_num, 52, hidden_dim)  # 初始细胞状态

        self.block = nn.Sequential(
                                    nn.Linear(3328, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(),
                                    nn.Linear(1024, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 64),
                                    )

    def forward(self, x):
        x = x.to('cuda')
        x = self.embedding(x) # [4019, 52] -> [4019, 52, 100]
        self.h0 = self.h0.to('cuda')
        self.c0 = self.c0.to('cuda')
        output, (hn, cn) = self.lstm(x, (self.h0, self.c0))  # 输出[4019, 52, 64]

        output = output.reshape(output.shape[0], -1) # [4019, 3328]
        #         print(output.shape,hn.shape)
        return self.block(output)

