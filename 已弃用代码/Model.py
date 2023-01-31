import torch
import torch.nn as nn
import numpy as np

# class Encoder(nn.Module):
#     def __init__(self, vocab_size):
#         super(Encoder, self).__init__()
#         self.input_size = vocab_size
#         self.embedding_size = 256
#         self.hidden_size = 256
#         self.layers = 3
#         self.embedding = nn.Embedding(self.input_size, self.embedding_size)
#         self.lstm = nn.LSTM(self.embedding, self.hidden_size, self.layers)
#
#     def forward(self, x):
#         embedding = self.embedding(x)
#         outputs, (hidden_state, cell_state) = self.lstm(embedding)
#
# class Decoder(nn.Module):
#     def __init__(self, vocab_size, output_size):
#         super(Decoder, self).__init__()
#         self.input_size = vocab_size
#         self.embedding_size = 256
#         self.output_size = output_size
#         self.hidden_size = 256
#         self.layers = 3
#         # self.dropout = nn.Dropout()
#         self.max_length = 26
#         self.embedding = nn.Embedding(self.input_size, self.embedding_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.lstm = nn.LSTM(self.embedding, self.hidden_size, self.layers)
#         self.fc = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, x, hidden_state, cell_state):
#         x = x.unsqueeze(0)
#         embedding = self.embedding(x)
#         outputs, (hidden_state, cell_state) = self.lstm(embedding, (hidden_state, cell_state))
#
#
#         predictions = self.fc(outputs)
#         return predictions, hidden_state, cell_state
#
#
# class Seq2Seq(nn.Module):
#     def __init__(self):
#         super(Seq2Seq, self).__init__()
#         self.encoder = Encoder
#         self.decoder = Decoder
#
#     def forward(self, )


class EncoderRNN(nn.Module):
    def __init__(self, input_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = 256
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=3)

    def forward(self, input, hidden):
        # seq_len, batch_size = inputs.size()
        embedbed = self.embedding(input).view(-1, 600, 256)

        output = embedbed

        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(3, 600, self.hidden_size).to('cuda')

class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = 256
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = 24

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=3)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(-1, 600, 256)
        embedded = self.dropout(embedded)
        attn_weights = nn.functional.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        print(attn_applied.shape)
        print(embedded.shape)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)

        output = nn.functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(3, 600, self.hidden_size).to('cuda')
