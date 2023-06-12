import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, output_dim):
        super(TextCNN, self).__init__()
        self.visualization = False

        vocab_size = 24
        dim_embedding = 100
        filter_num = 64
        filter_sizes = [1, 2, 4, 8, 16]

        self.embedding = nn.Embedding(vocab_size, dim_embedding)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, dim_embedding)) for fsz in filter_sizes])
        self.linear = nn.Linear(filter_num*len(filter_sizes), output_dim)

    def forward(self, x):
        # print(x)
        # print(x.shape)
        x = self.embedding(x)  # 外面处理了 [4019, 52, 100]
        # print(x.shape)
        x = x.view(x.size(0), 1, x.size(1), -1)  # [4019, 1, 52, 100]
        x = [F.relu(conv(x)) for conv in self.convs]  # list 5 [4019, 64, 52, 100]
        # print(len(x),x[0].shape,x[1].shape,x[2].shape)  # 5 torch.Size([4019, 64, 52, 1]) torch.Size([4019, 64, 51, 1]) torch.Size([4019, 64, 49, 1])
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        # print(len(x),x[0].shape,x[1].shape,x[2].shape)  # 5 torch.Size([4019, 64, 1, 1]) torch.Size([4019, 64, 1, 1]) torch.Size([4019, 64, 1, 1])
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        # print(len(x), x[0].shape)  # 5 torch.Size([4019, 64])
        embedding = torch.cat(x, 1)
        # print(len(embedding), embedding[0].shape)  # 4019 torch.Size([320])
        embedding = self.linear(embedding)
        # print(len(embedding), embedding[0].shape)  # 4019 torch.Size([320])
        # 1/0
        return embedding
