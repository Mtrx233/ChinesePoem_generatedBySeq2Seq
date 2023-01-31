import json
import re
import numpy as np
import os


def parse_raw_data(data_path, category, author, constrain):
    def sentence_parse(para):
        result, number = re.subn("（.*）", "", para)
        result, number = re.subn("{.*}", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("[\]\[]", "", result)

        r = ""
        for s in result:
            if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
                r += s;

        r, number = re.subn("。。", "。", r)

        return r

    def handle_json(file):
        rst = []
        data = json.loads(open(file, encoding='utf-8').read())
        for poetry in data:
            pdata = ""
            if author is not None and poetry.get("author") != author:
                continue
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split("[， ！ 。]", s)
                for tr in sp:
                    if constrain is not None and len(tr) != constrain \
                            and len(tr) != 0:
                        flag = False
                        break
                    if flag:
                        break
            if flag:
                continue

            for sentence in poetry.get("paragraphs"):
                pdata += sentence
            pdata = sentence_parse(pdata)
            if pdata != "" and len(pdata) > 1:
                rst.append(pdata)
        return rst

    data = []
    for filename in os.listdir(data_path):
        if filename.startswith(category):
            data.extend(handle_json(data_path + filename))
    return data


def pad_seq(seq, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    if not hasattr(seq, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in seq:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(seq)
    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = tuple()
    for s in seq:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(seq):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def get_data(config):
    data = parse_raw_data(
        config.data_path,
        config.category,
        config.author,
        config.constrain
    )

    chars = {c for line in data for c in line}
    char_to_ix = {char: ix for ix, char in enumerate(chars)}
    char_to_ix['<EOP>'] = len(char_to_ix)
    char_to_ix['<START>'] = len(char_to_ix)
    char_to_ix['</s>'] = len(char_to_ix)

    ix_to_char = {ix: char for char, ix in list(char_to_ix.items())}

    for i in range(0, len(data)):
        data[i] = ['<START>'] + list(data[i]) + ['<EOP>']

    data_id = [[char_to_ix[w] for w in line] for line in data]

    pad_data = pad_seq(
        data_id,
        maxlen=config.poetry_max_len,
        padding='pre',
        truncating='post',
        value=len(char_to_ix) - 1
    )

    np.savez_compressed(config.processed_data_path,
                        data=pad_data,
                        word2ix=char_to_ix,
                        ix2word=ix_to_char)

    return pad_data, char_to_ix, ix_to_char


def get_most_frequent(data, num):
    from collections import Counter
    result = Counter()
    for i in data:
        temp = Counter(i)
        result += temp
    list = result.most_common(num + 2)
    most_list = []
    for i in list:
        most_list.append(i[0])
    most_list = most_list[2:]
    print(most_list)
    frequency_list = []
    for i in most_list:
        frequency_list.append(result[i])
        print(i, result[i])
    return most_list, frequency_list


def getKeyWord(str,keysSet):
    resultList = []
    strList = list(str)
    for word in strList:
        if word in keysSet:
            resultList.append(word)
    return resultList


def dataSetHandle(dataSet,keySet):
    keyList = []
    poemList = []
    for poem in dataSet:
        poetry = poem
        senList = poem.replace("，","。").split("。",3)
        listOne = getKeyWord(senList[0],keySet)
        listTwo = getKeyWord(senList[1],keySet)
        listThree = getKeyWord(senList[2],keySet)
        listFour = getKeyWord(senList[3],keySet)
        listResult = dataSetList(listOne,listTwo,listThree,listFour)
        for word in listResult:
            keyList.append(word)
            poemList.append(poetry)
    return keyList,poemList



def dataSetList(listOne, listTwo, listThree, listFour):
    listFront = []
    listBehind = []
    for strOne in listOne:
        for strTwo in listTwo:
            str = list(strOne) + list(strTwo)
            listFront.append(str)

    for strThree in listThree:
        for strFour in listFour:
            str = list(strThree) + list(strFour)
            listBehind.append(str)

    result = []

    for strFront in listFront:
        for strBehind in listBehind:
            str = strFront + strBehind
            result.append(str)

    return result



def get_input(config):
    from zhconv import convert
    data = parse_raw_data(
        config.data_path,
        config.category,
        config.author,
        config.constrain
    )
    data_list = []
    for s in data:
        l = s.replace("，", "。").split("。")
        temp_num = 0
        temp_str = ""
        for i in l:
            if temp_num % 2 == 0:
                temp_str = temp_str + i + "，";
            else:
                temp_str = temp_str + i + "。";
            temp_num += 1
            if len(temp_str) == 24 and temp_num == 4:
                data_list.append(convert(temp_str, 'zh-cn'))
            if temp_num == 4:
                temp_str = ""
                temp_num = 0

    chars = {c for line in data_list for c in line}
    char_to_ix = {char: ix for ix, char in enumerate(chars)}
    char_to_ix['<EOP>'] = len(char_to_ix)
    char_to_ix['<START>'] = len(char_to_ix)
    char_to_ix['</s>'] = len(char_to_ix)

    ix_to_char = {ix: char for char, ix in list(char_to_ix.items())}
    #
    # for i in range(0, len(data_list)):
    #     data[i] = ['<START>'] + list(data[i]) + ['<EOP>']

    data_id = [[char_to_ix[w] for w in line] for line in data]

    key_list, frequency_list = get_most_frequent(data_list, 200)
    # print(key_list)
    print(frequency_list)

    key_list = ['不', '人', '山', '无', '日', '风', '云', '一', '有', '何', '天', '中', '水', '来', '时', '月', '生', '上', '心', '春', '为',
                '自', '相', '花', '长', '秋', '此', '如', '清', '行', '归', '知', '白', '君', '年', '空', '见', '高', '去', '在', '下', '远',
                '夜', '江', '未', '客', '多', '寒', '明', '里', '道', '门', '得', '子', '出', '青', '路', '落', '我', '雨', '朝', '入', '事',
                '思', '草', '与', '三', '千', '金', '南', '深', '地', '声', '处', '流', '色', '树', '城', '已', '新', '是', '万', '今', '开',
                '家', '玉', '林', '可', '飞', '还', '别', '东', '前', '石', '烟', '独', '同', '光', '闻', '谁', '方', '游', '尽', '望', '将',
                '复', '成', '回', '欲', '海', '马', '酒', '西', '应', '难', '重', '古', '闲', '身', '雪', '阳', '从', '气', '情', '亦', '书',
                '老', '衣', '非', '鸟', '更', '言', '尘', '名', '能', '香', '愁', '向', '意', '分', '看', '叶', '当', '然', '发', '所', '旧',
                '歌', '松', '满', '平', '余', '外', '幽', '北', '大', '离', '起', '犹', '过', '竹', '故', '作', '龙', '华', '随', '孤', '后',
                '黄', '安', '晚', '间', '暮', '阴', '野', '坐', '临', '初', '莫', '露', '终', '仙', '国', '文', '世', '诗', '似', '几', '楼',
                '台', '百', '岁', '公', '泉', '若', '吟', '五', '芳', '岂', '汉']
    list1, list2 = dataSetHandle(data_list, key_list)
    return list1, list2, key_list, frequency_list, char_to_ix, ix_to_char


def get_input1(config):
    from zhconv import convert
    data = parse_raw_data(
        config.data_path,
        config.category,
        config.author,
        config.constrain
    )
    data_list = [convert(x, 'zh-cn') for x in data if len(x)==24]

    chars = {c for line in data_list for c in line}
    char_to_ix = {char: ix for ix, char in enumerate(chars)}
    char_to_ix['<EOP>'] = len(char_to_ix)
    char_to_ix['<START>'] = len(char_to_ix)
    char_to_ix['</s>'] = len(char_to_ix)

    ix_to_char = {ix: char for char, ix in list(char_to_ix.items())}

    from Data_preserved import Data_preserved as Data

    key_list = Data.key_list_small
    frequency_list = Data.frequency_list_small
    list1, list2 = dataSetHandle(data_list, key_list)
    return list1, list2, key_list, frequency_list, char_to_ix, ix_to_char



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




if __name__ == '__main__':
    from Config import Config
    config = Config()
    get_input1(config)
    # from Data_preserved import Data_preserved as Data
    # print(Data.key_list)
    # print(Data.frequency_list)
    # list1, list2, key_list, frequency_list = get_input1(config)
    # print(frequency_list)
    # print(list2)
    # print(key_list)

    # pad_data, char_to_ix, ix_to_char = get_data(config)
    # for l in pad_data[:10]:
    #     print(l)
    #
    # n = 0
    # for k, v in char_to_ix.items():
    #     print(k, v)
    #     if n > 10:
    #         break
    #     n += 1
    #
    # n = 0
    # for k, v in ix_to_char.items():
    #     print(k, v)
    #     if n > 10:
    #         break
    #     n += 1
