from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn
import torch.nn.functional as F

class LSTMMultiheadAttention(nn.Module):
    def __init__(self, config):
        super(LSTMMultiheadAttention, self).__init__()
        self.datatype = config.datatype
        self.attention = config.attention
        self.hidden_size = config.hidden_size
        if self.attention:
            self.force = config.attentionforce
        self.normal = config.normal
        if config.bidirect:
            self.hidden_size = config.hidden_size * 2
        if self.datatype in ['word','wordprd']:
            self.embedding = nn.Embedding(config.len_voca, config.embed,max_norm=3)
        self.batchnorm = nn.BatchNorm1d(self.hidden_size)
        self.layernorm = nn.LayerNorm(
            normalized_shape=self.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)  # 独立 Dropout 层
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,dropout=config.dropout,batch_first=True,bidirectional=config.bidirect)
        if self.attention:
            self.multihead = nn.MultiheadAttention(self.hidden_size,config.nhead,dropout=config.dropout,bias=True,add_bias_kv=False)
            transformer_encoder_layer = TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=config.nhead,  # Transformer头数
                dim_feedforward=self.hidden_size*4,  # 前馈网络维度
                dropout=config.dropout
            )
            self.transformer_encoder = TransformerEncoder(
                transformer_encoder_layer,
                num_layers=config.head_layer  # Transformer层数
            )
        self.fc = nn.Linear(self.hidden_size, config.num_classes)

    def forward(self, x):
        if self.datatype in ['word','wordprd']:
            out = self.embedding(x)
        else:
            out = x
        out, _ = self.lstm(out)              #  输出batchsize,seqlen,hiddensize
        out = self.dropout(out)
        if self.normal:
            out = out.permute(0,2,1)             #  归一化输入batchsize,hiddensize,seqlen
            out = self.batchnorm(out)
            out = out.permute(0,2,1)             #  回归batchsize,seqlen,hiddensize
            # out = self.layernorm(out)
        if self.attention:
            out = out.permute(1, 0, 2)
            if self.force:
                out = self.transformer_encoder(out)
            else:
                out,_ = self.multihead(out,out,out)   #  输出的是 seqlen,batchsize,hiddensize
            # if self.normal:
            #     # out = out.permute(1,2,0)          #  输入的是 batchsize,hiddensize,seqlen
            #     out = self.layernorm(out)
            #     # out = out.permute(2,0,1)
            out = out.permute(1, 0, 2)
        output = self.fc(out[:, -1, :])
        return output