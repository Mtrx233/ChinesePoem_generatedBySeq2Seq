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
        self.gru = nn.GRU(256, 256, num_layers=3)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = 256
        self.fc = nn.Linear(self.hidden_size, input_size)

    def forward(self, decoder_input, hidden, encoder_output):
        decoder_input = decoder_input.view(1, -1)
        decoder_input = decoder_input.transpose(0, 1)  # src = [batch_size, src_len]
        # embedded = [src_len, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(decoder_input)).transpose(0, 1)

        decoder_output, hidden = self.gru(embedded, hidden)
        predictions = self.fc(decoder_output).squeeze(0)
        return decoder_output, hidden, predictions

class Seq2Seq(nn.Module):
    def __init__(self,vocab_size,ix_to_char,is_test=False):
        super(Seq2Seq, self).__init__()
        self.vocab_size= len(ix_to_char)
        self.encoder = Encoder(self.vocab_size, 0.1)
        self.decoder = Decoder(self.vocab_size, 0.1)
        self.is_test = is_test
        self.ix_to_char=ix_to_char

    def forward(self, source):
        str = ""
        encoder_input = source[0]
        decoder_input = source[1]
        encoder_input = encoder_input.long().transpose(1, 0).contiguous().to('cuda')
        decoder_input = decoder_input.long().transpose(1, 0).contiguous().to('cuda')
        input_, decoder_input, target = encoder_input, decoder_input[:-1, :], decoder_input[1:, :]
        encoder_hidden = self.encoder.initHidden().to('cuda')
        encoder_output, hidden = self.encoder(encoder_input, encoder_hidden)
        decoder_input = decoder_input[0]
        predict = torch.zeros(25, 60, self.vocab_size)
        for di in range(len(target)):
            decoder_output, hidden, prediction = self.decoder(decoder_input, hidden, encoder_output)
            predict[di] = prediction
            decoder_input = target[di]
            best_guess = prediction.argmax(1)
            str += self.ix_to_char[best_guess[0].item()]
        predict = predict.reshape(-1, predict.shape[2]).to('cuda')
        target = target.reshape(-1)
        if self.is_test ==True:
            return str
        else:
            return predict,target,str


