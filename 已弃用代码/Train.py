import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from DataProcess_1 import *
from Config import Config
import os
from Data_preserved import Data_preserved
from Model import EncoderRNN, AttnDecoderRNN
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class TrainModel(object):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.config = Config()

    def run(self):
        # data, char_to_ix, ix_to_char = get_data(self.config)
        # vocab_size = len(char_to_ix)
        # print('样本数：%d' % len(data))
        # print('词典大小： %d' % vocab_size)
        writer = SummaryWriter('history')
        cnt = 0

        encoder_data, decoder_data, key_list, frequency_list, char_to_ix, ix_to_char = get_input1(self.config)
        vocab_size = len(char_to_ix)
        # print('样本数：%d' % len(data))
        print('词典大小： %d' % vocab_size)
        for i in range(len(encoder_data)):
            for j in range(len(encoder_data[i])):
                encoder_data[i][j] = char_to_ix[encoder_data[i][j]]

        for i in range(len(decoder_data)):
            decoder_data[i] = ['<START>'] + list(decoder_data[i])\
            + ['<EOP>']
            for j in range(len(decoder_data[i])):
                decoder_data[i][j] = char_to_ix[decoder_data[i][j]]

######################################################################

        # data = torch.from_numpy(data)
        #
        encoder_data = torch.tensor(encoder_data)
        decoder_data = torch.tensor(decoder_data)
        data = []
        for i in range(len(encoder_data)):
            data.append([encoder_data[i], decoder_data[i]])

        # data = torch.from_numpy(data)

        dataloader = Data.DataLoader(
            data,
            shuffle=True,
            num_workers=1,
            batch_size=self.config.batch_size,
            drop_last=True
        )

        encoder = EncoderRNN(vocab_size).to('cuda')
        decoder = AttnDecoderRNN(vocab_size).to('cuda')
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=Config.learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=Config.learning_rate)

        loss_function = nn.CrossEntropyLoss()


        for epoch in range(Config.EPOCH):
            for step, x in enumerate(tqdm(dataloader)):
                cnt += 1
                # 1.处理数据
                # x: (batch_size,max_len) ==> (max_len, batch_size)
                encoder_input = x[0]
                decoder_input = x[1]
                encoder_input = encoder_input.long().transpose(1, 0).contiguous().to('cuda')
                decoder_input = decoder_input.long().transpose(1, 0).contiguous().to('cuda')
                # input,target:  (max_len, batch_size-1)

                input_, decoder_input,target = encoder_input, decoder_input[:-1, :], decoder_input[1:, :]

                # decoder_input = decoder_input.view(-1)
                # target = target.view(-1)
                # 初始化hidden为(c0, h0): ((layer_num， batch_size, hidden_dim)，(layer_num， batch_size, hidden_dim)）

                encoder_hidden = encoder.initHidden()
                encoder_outputs = torch.zeros(24, encoder.hidden_size).to('cuda')

                for ei in range(len(encoder_input)):
                    encoder_output, encoder_hidden = encoder(encoder_input[ei],
                                                             encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0, 0]

                decoder_hidden = encoder_hidden
                decoder_input = decoder_input[0]

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                str = ""
                str_1 = ""

                use_teacher_forcing = True if random.random() < 0.5 else False
                if use_teacher_forcing:
                    for di in range(len(target)):
                        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                    decoder_hidden,
                                                                                    encoder_outputs)
                        loss = loss_function(decoder_output, target[di])
                        decoder_input = target[di]
                        top_index = decoder_output.data[0].topk(1)[1][0].item()

                        str += ix_to_char[top_index]
                        str_1 += ix_to_char[target[di][0].item()]
                else:
                    for di in range(len(target)):
                        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                    decoder_hidden,
                                                                                    encoder_outputs)
                        topv, topi = decoder_output.topk(1)
                        decoder_input = topi.squeeze()
                        top_index = decoder_output.data[0].topk(1)[1][0].item()

                        str += ix_to_char[top_index]
                        str_1 += ix_to_char[target[di][0].item()]

                        loss = loss_function(decoder_output, target[di])
                        # if decoder_input.equal(target[-1]):
                        #     break


                writer.add_scalar("Training loss", loss.item(), global_step=cnt)
                print(loss)
                print(str)
                # print(str_1)

                loss.backward()
                decoder_optimizer.step()
                encoder_optimizer.step()
        torch.save(encoder, "encoder.pth")
        torch.save(decoder, "decoder.pth")
                # print(loss)
                # return loss.item() / target_length

                # # 2.前向计算
                # # print(input.size(), hidden[0].size(), target.size())
                # output, _ = model(input_, hidden)
                # loss = criterion(output, target)  # output:(max_len*batch_size,vocab_size), target:(max_len*batch_size)
                #
                # # 反向计算梯度
                # loss.backward()
                #
                # # 权重更新
                # optimizer.step()
                #
                # if step == 0:
                #     print('epoch: %d,loss: %f' % (epoch, loss.data))




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