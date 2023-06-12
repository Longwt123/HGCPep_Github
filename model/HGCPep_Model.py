import torch
import torch.nn as nn
from dhg.models import HGNN, HGNNP, HyperGCN, HNHN, UniGCN, UniGAT, UniSAGE, UniGIN
from model.embedding_module.TextCNN import TextCNN
from model.embedding_module.cnn import CNNnet
from model.embedding_module.gru import GRUModel
from model.embedding_module.lstm import LSTMModel
from model.embedding_module.LSTMwithAttention import LSTMAttention
from model.embedding_module.RNN_CNN import RNN_CNN
from model.embedding_module.ESM2 import ESMTextCNN
from model.embedding_module.PeptideBERT import PeptideBERT
import esm


"""
HGCPep 模型
"""

aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
           'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
           'W': 20, 'Y': 21, 'V': 22, 'X': 23}


class HGCPep_Model(nn.Module):
    def __init__(self, config, num_classes=None):
        super().__init__()
        self.config = config['HGCPep_model']
        self.frontPart = config['frontPart']
        self.latterPart = config['latterPart']
        self.lastPart = config['lastPart']
        self.input_dim = self.config['input_dim']
        self.mid_dim = self.config['mid_dim']
        self.hidden_units = self.config['hidden_units']
        self.draw_class = config['draw_data']['draw_class']
        if num_classes is None:
            num_classes = [2 for i in range(config['class_num'])]
        self.num_classes = num_classes

        '''frontPart'''
        # self.embedding = nn.Embedding(24, self.input_dim)
        match self.frontPart:
            case "textCNN":
                self.textCNN = TextCNN(self.hidden_units)
            case "GRU":
                self.gru = GRUModel(24, self.input_dim, self.hidden_units)
            case "LSTMwithAttention":
                self.lstmAttention = LSTMAttention(self.input_dim, self.hidden_units)
            case "lstm":
                self.lstm = LSTMModel(24, self.input_dim, self.hidden_units)
            case "RNN_CNN":
                self.rnncnn = RNN_CNN(self.input_dim, self.hidden_units)
            case "CNN":
                self.cnn = CNNnet()


        '''latterPart'''
        if self.latterPart == 'HGNNP':
            self.hgnnp = HGNNP(self.hidden_units, self.mid_dim, 64, use_bn=True)
        if self.latterPart == 'HGNN':
            self.hgnn = HGNN(self.hidden_units, self.mid_dim, 64, use_bn=True)
        if self.latterPart == 'HyperGCN':
            self.hyperGCN = HyperGCN(self.hidden_units, self.mid_dim, 64)
        if self.latterPart == 'HNHN':
            self.hnhn = HNHN(self.hidden_units, self.mid_dim, 64, use_bn=True)
        if self.latterPart == 'UniGCN':
            self.unigcn = UniGCN(self.hidden_units, self.mid_dim, 64, use_bn=True)
        if self.latterPart == 'UniGAT':
            self.unigat = UniGAT(self.hidden_units, self.mid_dim, 64, 8, use_bn=True)
        if self.latterPart == 'UniSAGE':
            self.unisage = UniSAGE(self.hidden_units, self.mid_dim, 64, use_bn=True)
        if self.latterPart == 'UniGIN':
            self.unigin = UniGIN(self.hidden_units, self.mid_dim, 64, use_bn=True)

        '''otherPart'''
        # self.ensembleNorm = self.config['ensembleNorm']
        # self.ensembleMode = self.config['ensembleMode']

        """FullyConnectedLayer"""
        for index, num_class in enumerate(num_classes):
            setattr(self, "FullyConnectedLayer1_" + str(index), nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=32),
                # nn.Dropout(p=0.5)
            ))
            setattr(self, "FullyConnectedLayer2_" + str(index), nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=32, out_features=2),
                # nn.Dropout(p=0.5)
            ))

    def forward(self, x, A):
        """################  frontPart  ###################"""
        representationX = x
        match self.frontPart:
            case "textCNN":
                representationX = self.textCNN(representationX)
            case "GRU":
                representationX = self.gru(representationX)
            case "LSTMwithAttention":
                representationX = self.lstmAttention(representationX)
            case "lstm":
                representationX = self.lstm(representationX)
            case "RNN_CNN":
                representationX = self.rnncnn(representationX)
            case "CNN":
                representationX = self.cnn(representationX)
        # print(representationX.shape)  # [4019, 64]

        """################  latterPart  ###################"""
        representationY = None
        match self.latterPart:
            case "HGNNP":
                representationY = self.hgnnp(representationX, A)
            case "HGNN":
                representationY = self.hgnn(representationX, A)
            case "HyperGCN":
                representationY = self.hyperGCN(representationX, A)
            case "HNHN":
                representationY = self.hnhn(representationX, A)
            case "nothing":
                representationY = representationX
            case "UniGCN":
                representationY = self.unigcn(representationX, A)
            case "UniGAT":
                representationY = self.unigat(representationX, A)
            case "UniSAGE":
                representationY = self.unisage(representationX, A)
            case "UniGIN":
                representationY = self.unigin(representationX, A)



        out_embedding = representationY  # 画图用

        representation = representationY
        outs = list()
        dir(self)
        for index, num_class in enumerate(self.num_classes):
            fun1 = eval("self.FullyConnectedLayer1_" + str(index))
            fun2 = eval("self.FullyConnectedLayer2_" + str(index))
            out_temp = fun1(representation.reshape(representation.shape[0], -1))
            out = fun2(out_temp)
            if index == self.draw_class:  # 取对应类的输出画图
                out_embedding = out_temp  # out_temp
            outs.append(out)

        return outs, out_embedding
