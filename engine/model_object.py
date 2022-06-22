import json
import copy


class ConfigModel(object):
    def __init__(self):
        self.use_gram = None
        self.gram_hidden_size = None
        self.gram_dropout = None
        self.use_chars = None
        self.char_max_word_length = None
        self.char_embedding_dim = None
        self.char_function_hidden_size = None
        self.char_dropout = None
        self.char_function_output_size = None
        self.use_word_embeddings = None
        self.word_embedding_dropout = None
        self.word_max_count = None
        self.use_trained_char_embeddings = None
        self.char_model_config_path = None
        self.char_model_weights_path = None
        self.rnn_input_size = None
        self.rnn_hidden_size = None
        self.rnn_n_layers = None
        self.rnn_dropout = None
        self.rnn_bidirectional = None
        self.dense_size = None
        self.dense_dropout = None
        self.use_crf = None
        self.use_pos_lm = None
        self.use_word_lm = None
        if self.use_word_lm:
            assert not self.use_word_embeddings

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            d = copy.deepcopy(self.__dict__)
            f.write(json.dumps(d, sort_keys=True, indent=4) + "\n")

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            d = json.loads(f.read())
            self.__dict__.update(d)


class ConfigTrain(object):
    def __init__(self):
        self.eval_model_config_path = None
        self.eval_model_weights_path = None
        self.train_model_config_path = None
        self.train_model_weights_path = None
        self.gram_dict_input = None
        self.gram_dict_output = None
        self.word_vocabulary = None
        self.char_set_path = None
        self.rewrite_model = True
        self.external_batch_size = None
        self.batch_size = None
        self.sentence_len_groups = None
        self.val_part = None
        self.epochs_num = None
        self.dump_model_freq = None
        self.random_seed = None

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            d = copy.deepcopy(self.__dict__)
            f.write(json.dumps(d, sort_keys=True, indent=4) + "\n")

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            d = json.loads(f.read())
            self.__dict__.update(d)
