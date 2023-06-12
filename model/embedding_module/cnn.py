import torch
import torch.nn as nn

# Define the one-dimensional convolutional neural network
class CNNnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=4, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(104, 64)

    def forward(self, x):
        x = x.to('cuda')
        x = x.type(torch.cuda.FloatTensor)
        x = x.view([4019, 1, -1])
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        x = torch.cat((y1, y2), dim=2)
        x = self.relu(x)
        x = self.fc(x)
        x = torch.squeeze(x)
        return x
