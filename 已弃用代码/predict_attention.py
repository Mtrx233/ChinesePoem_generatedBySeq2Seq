import numpy as np
from gensim.models import word2vec
import torch
from Model_attention import  Encoder ,Decoder

epoch = 4
def get_key_word(key_list_small,sentence,model):
    '''
    提取关键字
    :param key_list_small: 关键字列表
    :param sentence: 要预测的句子
    :param model: word2vec模型
    :return: 返回四个关键词
    '''
    from numpy import random
    key =[]

    for i in key_list_small:
        if i in sentence and len(key)<4:
            key.append(i)

    if len(key) == 1:
        topn = model.wv.most_similar(key[0], topn=3)
        for i in topn:
            key.append(i[0])

    if len(key) ==2:
        top_one = model.wv.most_similar(key[0], topn=3)
        top_two = model.wv.most_similar(key[1], topn=3)
        key.append(top_one[random.randint(0,2)][0])
        key.append(top_two[random.randint(0,2)][0])
    if len(key)==3:
        print(key)
        top_one = model.wv.most_similar(key[0], topn=3)
        top_two = model.wv.most_similar(key[1], topn=3)
        top_three = model.wv.most_similar(key[2], topn=3)

        temp = [v for v in top_one if v in top_two and v in top_three]
        if len(temp)>0:
            key.append(temp[random.randint(0,len(temp)-1)])
        else:
            x = random.randint(1,3)
            if x ==1 :
                key.append(top_one[random.randint(0,2)][0])
            elif x==2:
                key.append(top_two[random.randint(0, 2)][0])
            else:
                key.append(top_three[random.randint(0, 2)][0])
    return key

def get_input(str_input):

    # # 引入日志配置
    key_list_small = ['不', '人', '无', '风', '一', '山', '日', '花', '来', '何', '月', '春', '有', '中', '水', '上', '心', '时', '知',
                      '秋',
                      '夜', '自', '见', '云', '相', '如', '江', '君', '天', '里', '年', '为', '长', '是', '生', '处', '白', '去', '明',
                      '下',
                      '空', '归', '行', '得', '未', '多', '在', '今', '千', '寒', '青', '此', '落', '客', '家', '飞', '南', '莫', '声',
                      '草',
                      '清', '金', '子', '高', '道', '还', '欲', '谁', '别', '将', '独', '路', '城', '出', '尽', '门', '看', '树', '三',
                      '事',
                      '万', '朝', '入', '开', '色', '可', '远', '雨', '向', '头', '玉', '回', '叶', '愁', '意', '前', '流', '作', '似',
                      '与',
                      '深', '东', '应', '更', '阳', '烟', '我', '望', '满', '酒', '西', '到', '复', '犹', '香', '地', '衣', '闻', '新',
                      '已',
                      '黄', '须', '马', '同', '雪', '思', '石', '成', '情', '红', '非', '古', '起', '林', '若', '能', '身', '故', '重',
                      '柳',
                      '泪', '枝', '边', '暮', '大', '光', '当', '平', '游', '难', '过', '歌', '好', '北', '从', '逢', '间', '尘', '楼',
                      '言',
                      '鸟', '发', '海', '问', '绿', '方', '老', '亦', '乡', '照', '孤', '闲', '竹', '却', '后', '外', '怜', '百', '临',
                      '离',
                      '宫', '郎', '两', '露', '几', '坐', '半', '霜', '断', '十', '共', '书', '醉', '吹', '语', '仙', '溪', '影', '梦',
                      '河',
                      '正', '旧']

    model = word2vec.Word2Vec(sentences=key_list_small, vector_size=200, window=7, min_count=1)
    key_list = get_key_word(key_list_small, str_input, model)

    key_list=['明','月','不','千']
    print(key_list)

    char_to_ix  = np.load('result/char_to_ix_attention_{}.npy'.format(epoch), allow_pickle=True).item()
    ix_to_char = np.load('result/ix_to_char_attention_{}.npy'.format(epoch), allow_pickle=True).item()

    key_ix = []
    for key in key_list:
        key_ix.append(char_to_ix[key])

    keys = []

    for i in key_list_small:
        if i in str_input:
            keys.append(i)
    str_input = []
    for char in keys:
        str_input.append(char_to_ix[char])

    decoder_input = [char_to_ix["<START>"]]

    return str_input, key_ix, decoder_input, char_to_ix, ix_to_char


def predict(str_input):

    str_input, key_list, decoder_input, char_to_ix, ix_to_char = get_input(str_input)

    print(key_list)

    encoder_static = torch.load("result/encoder_attention_{}.pth".format(epoch)).to('cpu').state_dict()
    decoder_static = torch.load("result/decoder_attention_{}.pth".format(epoch)).to('cpu').state_dict()
    #3752
    encoder =Encoder(3752,0).to('cpu')
    decoder = Decoder(3752, 0).to('cpu')
    encoder.load_state_dict(encoder_static)
    decoder.load_state_dict(decoder_static)

    encoder_input = torch.tensor(key_list)
    decoder_input = torch.tensor(decoder_input)

    encoder_input= encoder_input.view(4,-1)
    decoder_input= decoder_input.view(1,-1)
    # [seq_length, batch_size]

    encoder_hidden = torch.zeros(3, 1,256)
    print(encoder_input.shape)
    encoder_output, hidden = encoder(encoder_input, encoder_hidden)
    predict = torch.zeros(25, 1, len(char_to_ix))
    decoder_input = decoder_input[0]

    length = 25
    str = ""

    for di in range(length):
        decoder_output, hidden, prediction = decoder(decoder_input, hidden, encoder_output)

        # prediction = top_k_top_p_filtering(prediction, 512, 0.9)

        predict[di] = prediction
        decoder_input = torch.tensor([prediction.argmax(1)[0].item()])
        str += ix_to_char[prediction.argmax(1).item()]

    print(str)
    return str

predict("如色深为")






