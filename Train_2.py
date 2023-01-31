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

""" Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
"""
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    return logits


class TrainModel(object):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.config = Config()

    def run(self):
        writer = SummaryWriter('history_new')
        cnt = 0

        encoder_data, decoder_data, key_list, frequency_list, char_to_ix, ix_to_char = get_input1(self.config)
        vocab_size = len(char_to_ix)
        print('词典大小： %d' % vocab_size)
        for i in range(len(encoder_data)):
            for j in range(len(encoder_data[i])):
                encoder_data[i][j] = char_to_ix[encoder_data[i][j]]

        for i in range(len(decoder_data)):
            decoder_data[i] = ['<START>'] + list(decoder_data[i])\
            + ['<EOP>']
            for j in range(len(decoder_data[i])):
                decoder_data[i][j] = char_to_ix[decoder_data[i][j]]

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
            s = ""
            s_0 = ""
            for step, x in enumerate(tqdm(dataloader)):
                cnt += 1
                encoder_input = x[0]
                decoder_input = x[1]
                encoder_input = encoder_input.long().transpose(1, 0).contiguous().to('cuda')
                decoder_input = decoder_input.long().transpose(1, 0).contiguous().to('cuda')

                input_, decoder_input, target = encoder_input, decoder_input[:-1, :], decoder_input[1:, :]

                encoder_hidden = encoder.initHidden().to('cuda')
                encoder_output, hidden = encoder(encoder_input, encoder_hidden)

                decoder_input = decoder_input[0]
                predict = torch.zeros(25, 600, vocab_size).to('cuda')

                str = ""
                str_1 = ""
                str_0 = ""

                for i in range(len(encoder_input)):
                    str_0 += ix_to_char[encoder_input[i][0].item()]

                for di in range(len(target)):
                    decoder_output, hidden, prediction = decoder(decoder_input, hidden, encoder_output)

                    predict[di] = prediction
                    decoder_input = target[di]

                    best_guess = prediction.topk(512)[1]
                    for t in range(len(best_guess)):
                        best_guess[t] = random.choice(best_guess[t])

                    str += ix_to_char[best_guess[0][0].item()]
                    str_1 += ix_to_char[target[di][0].item()]

                predict = predict.reshape(-1, predict.shape[2])
                target = target.reshape(-1)
                decoder_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss = loss_function(predict, target)
                loss.backward()

                decoder_optimizer.step()
                encoder_optimizer.step()
                s_0 = str_0
                s = str


                writer.add_scalar("Training loss", loss.item(), global_step=cnt)
            print(s_0)
            print(s)


# def predict():
#     encoder = torch.load("encoder.pth")
#     decoder = torch.load("decoder.pth")
#
#     key_list_small = Data_preserved.key_list_small
#
#     model = word2vec.Word2Vec(sentences=key_list_small, vector_size=200, window=7, min_count=1)
#     sentence = "不"
#     l = get_key_word(key_list_small, sentence, model)





if __name__ == '__main__':
    obj = TrainModel()
    obj.run()