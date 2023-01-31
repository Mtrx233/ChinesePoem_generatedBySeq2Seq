import torch
import torch.nn as nn
import numpy as np
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, 256)
        # single layer, bi-direction GRU
        self.rnn = nn.GRU(256, 256, num_layers=3)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = 256

    def forward(self, src, hidden):
        '''
        :param src: [src_len, batch_size]
        :return:
        '''

        src = src.transpose(0, 1)  # src = [batch_size, src_len]
        # embedded = [src_len, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(src)).transpose(0, 1)

        # enc_output = [src_len, batch_size, hid_dim*num_directions]
        # enc_hidden = [n_layers * num_directions, batch_size, hid_dim]
        enc_output, enc_hidden = self.rnn(embedded, hidden)
        return enc_output, enc_hidden

    def initHidden(self):
        return torch.zeros(3, 60, self.hidden_size).to('cuda')

class Decoder(nn.Module):
    def __init__(self, input_size, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(input_size, 256)
        self.gru = nn.GRU(256*2, 256, num_layers=3)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = 256
        self.fc = nn.Linear(self.hidden_size, input_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size, 1)

    def forward(self, decoder_input, hidden, encoder_output):
        decoder_input = decoder_input.view(1, -1)
        decoder_input = decoder_input.transpose(0, 1)  # src = [batch_size, src_len]
        # embedded = [src_len, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(decoder_input)).transpose(0, 1)




        s = hidden[-1].unsqueeze(0).repeat(4, 1, 1)
        # s = [seq_len, batch_size, hidden_size]
        combined = self.attn_combine(self.attn(torch.cat((s, encoder_output), dim=2)))
        # combined = [seq_len, batch_size, 1]
        attn_weight = nn.functional.softmax(combined, dim=1).squeeze(2)
        # attn_weight = [seq_len, batch_size]
        attn_weight = attn_weight.transpose(0, 1)
        attn_weight = attn_weight.unsqueeze(1)
        # attn_weight = [batch_size, 1, seq_len]
        encoder_output = encoder_output.transpose(0, 1)
        # encoder_output = [batch_size, seq_len, hidden_size]
        context = torch.bmm(attn_weight, encoder_output)
        # context = [batch_size, 1, hidden_size]

        rnn_input = torch.cat((embedded.transpose(0, 1), context), dim=2)
        # rnn_input = [batch_size, 1, hidden_size + emb_size]

        decoder_output, hidden = self.gru(rnn_input.transpose(0, 1), hidden)
        predictions = self.fc(decoder_output).squeeze(0)
        return decoder_output, hidden, predictions

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_ratio = 0.5):
        batch_size = len(source[1])
        target_len = target.shape[0]




