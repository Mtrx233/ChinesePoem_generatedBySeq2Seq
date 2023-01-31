import datetime

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from DataProcess_1 import *
from Config import Config
import os
from Data_preserved import Data_preserved
from Model_new import Encoder, Decoder
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



class TrainModel(object):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.config = Config()

    def run(self):
        writer = SummaryWriter('history_new2/'+ datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        cnt = 0

        encoder_data, decoder_data, key_list, frequency_list, char_to_ix, ix_to_char = get_input1(self.config)
        vocab_size = len(char_to_ix)
        np.save("char_to_ix.npy",char_to_ix)
        np.save("ix_to_char.npy",ix_to_char)


        print('词典大小： %d' % vocab_size)
        for i in range(len(encoder_data)):
            for j in range(len(encoder_data[i])):
                encoder_data[i][j] = char_to_ix[encoder_data[i][j]]

        for i in range(len(decoder_data)):
            decoder_data[i] = ['<START>'] + list(decoder_data[i])\
            + ['<EOP>']
            data = []
            for j in range(len(decoder_data[i])):
                if j not in [0, 6, 12, 18, 24, 25]:
                    data.append(char_to_ix[decoder_data[i][j]])
                decoder_data[i][j] = char_to_ix[decoder_data[i][j]]
            decoder_data[i] = data

        encoder_data = torch.tensor(encoder_data)
        decoder_data = torch.tensor(decoder_data)
        data = []
        for i in range(len(encoder_data)):
            data.append([encoder_data[i], decoder_data[i]])
        dataloader = Data.DataLoader(
            data,
            shuffle=True,
            num_workers=1,
            batch_size=self.config.batch_size,
            drop_last=True
        )

        encoder = Encoder(vocab_size, 0.1).to('cuda')
        decoder = Decoder(vocab_size, 0.1).to('cuda')
        loss_function = nn.CrossEntropyLoss()
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=Config.learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=Config.learning_rate)

        for epoch in range(Config.EPOCH):

            for step, x in enumerate(tqdm(dataloader)):
                str = ""
                cnt += 1
                encoder_input = x[0]
                decoder_input = x[1]
                encoder_input = encoder_input.long().transpose(1, 0).contiguous().to('cuda')
                decoder_input = decoder_input.long().transpose(1, 0).contiguous().to('cuda')
                input_, decoder_input, target = encoder_input, decoder_input[:-1, :], decoder_input[1:, :]
                encoder_hidden = encoder.initHidden().to('cuda')
                encoder_output, hidden = encoder(encoder_input, encoder_hidden)

                decoder_input = decoder_input[0]
                predict = torch.zeros(19, 60, vocab_size).to('cuda')



                for di in range(len(target)):

                    decoder_output, hidden, prediction = decoder(decoder_input, hidden, encoder_output)
                    predict[di] = prediction
                    best_guess = prediction.argmax(1)
                    str += ix_to_char[best_guess[0].item()]
                    decoder_input = target[di]
                predict = predict.reshape(-1, predict.shape[2])
                target = target.reshape(-1)
                decoder_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss = loss_function(predict, target)
                loss.backward()
                decoder_optimizer.step()
                encoder_optimizer.step()

                if step % 100 ==0:
                    print(str)
                writer.add_scalar("Training loss", loss.item(), global_step=cnt)






if __name__ == '__main__':
    obj = TrainModel()
    obj.run()