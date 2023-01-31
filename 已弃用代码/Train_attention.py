import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from DataProcess_1 import *
from Config import Config
import os
from Data_preserved import Data_preserved
from Model_attention import Encoder, Decoder
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # if top_p > 0.0:
    #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #     cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    #
    #     # Remove tokens with cumulative probability above the threshold
    #     sorted_indices_to_remove = cumulative_probs > top_p
    #     # Shift the indices to the right to keep also the first token above the threshold
    #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #     sorted_indices_to_remove[..., 0] = 0
    #
    #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #     logits[indices_to_remove] = filter_value
    return logits

def perplexity():

    return None


class TrainModel(object):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.config = Config()

    def run(self):
        # data, char_to_ix, ix_to_char = get_data(self.config)
        # vocab_size = len(char_to_ix)
        # print('样本数：%d' % len(data))
        # print('词典大小： %d' % vocab_size)
        now = datetime.datetime.now()
        writer = SummaryWriter('history_new4/'+now.strftime("%Y-%m-%d%H%M%S"))
        cnt = 0

        encoder_data, decoder_data, key_list, frequency_list, char_to_ix, ix_to_char = get_input1(self.config)
        vocab_size = len(char_to_ix)
        # np.save('result/char_to_ix_attention_{}.npy'.format(1), char_to_ix)
        # np.save('result/ix_to_char_attention_{}.npy'.format(1), ix_to_char)
        print('词典大小： %d' % vocab_size)
        for i in range(len(encoder_data)):
            for j in range(len(encoder_data[i])):
                encoder_data[i][j] = char_to_ix[encoder_data[i][j]]

        for i in range(len(decoder_data)):
            decoder_data[i] = ['<START>'] + list(decoder_data[i])\
            + ['<EOP>']
            # data = []
            for j in range(len(decoder_data[i])):
                # if j not in [0, 6, 12, 18, 24, 25]:
                #     data.append(char_to_ix[decoder_data[i][j]])
                decoder_data[i][j] = char_to_ix[decoder_data[i][j]]
            # decoder_data[i] = data

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

        encoder = Encoder(vocab_size, 0.1).to('cuda')
        decoder = Decoder(vocab_size, 0.1).to('cuda')
        loss_function = nn.CrossEntropyLoss()
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)

        for epoch in range(Config.EPOCH):
            s = ""
            s_0 = ""
            for step, x in enumerate(tqdm(dataloader)):
                cnt += 1
                encoder_input = x[0]
                decoder_input = x[1]
                encoder_input = encoder_input.long().transpose(1, 0).contiguous().to('cuda')
                decoder_input = decoder_input.long().transpose(1, 0).contiguous().to('cuda')
                # input,target:  (max_len, batch_size-1)

                input_, decoder_input, target = encoder_input, decoder_input[:-1, :], decoder_input[1:, :]
                # encoder_input = tensor[4, 600], decoder_input = tensor[25, 600], target = tensor[25, 600]
                #                       [length, batch_size]

                encoder_hidden = encoder.initHidden().to('cuda')
                encoder_output, hidden = encoder(encoder_input, encoder_hidden)

                decoder_input = decoder_input[0]
                predict = torch.zeros(25, 60, vocab_size).to('cuda')

                str = ""
                str_1 = ""
                str_0 = ""

                for i in range(len(encoder_input)):
                    str_0 += ix_to_char[encoder_input[i][0].item()]



                for di in range(len(target)):
                    use_teacher_forcing = False if random.random() < 0.5 else True
                    decoder_output, hidden, prediction = decoder(decoder_input, hidden, encoder_output)

                    # prediction = top_k_top_p_filtering(prediction, 512, 0.9)

                    predict[di] = prediction
                    decoder_input = target[di]
                    if not use_teacher_forcing:
                        decoder_input = torch.tensor(prediction.argmax(1))

                    best_guess = prediction.argmax(1)
                    # best_guess = prediction.topk(512)[1]
                    # for t in range(len(best_guess)):
                    #     best_guess[t] = random.choice(best_guess[t])
                    # best_guess = prediction.random.choice()


                    # str += ix_to_char[best_guess[0][0].item()]
                    str += ix_to_char[best_guess[0].item()]
                    str_1 += ix_to_char[target[di][0].item()]

                predict = predict.reshape(-1, predict.shape[2])
                target = target.reshape(-1)
                decoder_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss = loss_function(predict, target)
                perplexity = torch.exp(loss)
                loss.backward()

                decoder_optimizer.step()
                encoder_optimizer.step()
                # print(str_0)
                s_0 = str_0
                # print(str)
                s = str
                # print(loss)


                if step % 100 == 0:
                    print(str)
                writer.add_scalar("Training loss", loss.item(), global_step=cnt)
            print(s_0)
            print(s)
            np.save('result/char_to_ix_attention_{}.npy'.format(epoch), char_to_ix)
            np.save('result/ix_to_char_attention_{}.npy'.format(epoch), ix_to_char)
            torch.save(encoder, "result/encoder_attention_{}.pth".format(epoch))
            torch.save(decoder, "result/decoder_attention_{}.pth".format(epoch))


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