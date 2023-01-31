import torch
import torch.nn as nn
import numpy as np
import random
from Config import Config


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_size, embedding_size, layers, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_dim, self.embedding_size)
        self.layers = layers
        self.rnn = nn.GRU(256, 256, num_layers=self.layers)

    '''
    :param src: [src_len, batch_size]
    :return:
    '''
    def forward(self, src, hidden):

        src = src.transpose(0, 1)
        embedded = self.dropout(self.embedding(src)).transpose(0, 1)

        enc_output, enc_hidden = self.rnn(embedded, hidden)
        return enc_output, enc_hidden

    def initHidden(self):
        return torch.zeros(self.layers, Config.Config.layers, self.hidden_size).to('cuda')

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.gru = nn.GRU(256, 256, num_layers=self.layers)
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
    def __init__(self, encoder, decoder, vocab_size, ix_to_char, char_to_ix):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char

    def forward(self, source, target, teacher_ratio=0.5):
        batch_size = len(source[1])
        target_len = target.shape[0]
        max_length = target_len - 1

        encoder_input, decoder_input, target = source, target[:-1, :], target[1:, :]

        encoder_hidden = self.encoder.initHidden().to('cuda')
        encoder_output, hidden = self.encoder(encoder_input, encoder_hidden)

        decoder_input = decoder_input[0]
        outputs = torch.zeros(max_length, batch_size, self.vocab_size).to('cuda')


        use_teacher_forcing = True if random.random() < teacher_ratio else False
        if use_teacher_forcing:
            for di in range(len(target)):
                decoder_output, hidden, prediction = self.decoder(decoder_input, hidden, encoder_output)
                outputs[di] = prediction
                decoder_input = target[di]
                # decoder_input = target[di]
                # best_guess = prediction.argmax(1)
                # str += ix_to_char[best_guess[0].item()]
                # str_1 += ix_to_char[target[di][0].item()]
        else:
            for di in range(len(target)):
                decoder_output, hidden, prediction = self.decoder(decoder_input, hidden, encoder_output)
                outputs[di] = prediction
                best_guess = prediction.argmax(1)
                decoder_input = self.ix_to_char

                if decoder_input.item() == "<EOP>":
                    break







