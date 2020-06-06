import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import math

class LSTM_set(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(LSTM_set, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        self.fc = nn.Linear(hidden_size* 2, 2)
        self.encoder = nn.LSTM(hidden_size , hidden_size, num_layers= 3, dropout=0.2, batch_first=True, bidirectional= True)

    def forward(self, input_data):
        out = F.relu(self.linear(input_data))
        decoder_set, _ = self.encoder(out)
        out = torch.sigmoid(self.fc(decoder_set))
        return out

class LSTM_pitch(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(LSTM_pitch, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        self.encoder = nn.LSTM(hidden_size , hidden_size, num_layers= 3, dropout=0.2, batch_first=True, bidirectional= True)
        self.fc = nn.Linear(hidden_size* 2, 1)

    def forward(self, input_data):
        out = F.relu(self.linear(input_data))
        decoder_pitch, _ = self.encoder(out)
        out = self.fc(decoder_pitch)

        return out
