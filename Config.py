class Config(object):
    data_path = "./chinese-poetry-master/json/"
    category = "poet.tang"
    author = None
    constrain = None
    poetry_max_len = 125
    sample_max_len = poetry_max_len - 1

    EPOCH = 100
    batch_size = 60
    embedding_dim = 256
    hidden_dim = 256
    layers = 2
    learning_rate = 0.5
    weight_decay =0.0001

    max_gen_len = 200
    sentence_max_len = 4
    prefix_words = '细雨鱼儿出,微风燕子斜。'
    start_words = '闲云潭影日悠悠'
    acrostic = False

    model_path = './model.pth'
    model_prefix = './tang'

    word_dict_path = 'wordDic'
    processed_data_path = "./tang.npz"

