import torch
import torch.nn as nn

"""
使用PyTorch实现的单个GRU单元的代码
结合两种信息
"""


class GRUModel(nn.Module):
    def __init__(self, vocab_size=24, emb_dim=100, hidden_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.2)

        self.block = nn.Sequential(
                                    nn.Linear(6912, 1024),
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
        output = x.permute(1, 0, 2)
        output, hn = self.gru(output)
        output = output.permute(1, 0, 2) # [4019, 52, 100]
        hn = hn.permute(1, 0, 2)
        output = output.reshape(output.shape[0], -1)
        hn = hn.reshape(output.shape[0], -1)
        output = torch.cat([output, hn], 1) # [4019, 6912]
        #         print(output.shape,hn.shape)
        return self.block(output)

