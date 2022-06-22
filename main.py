import json
from engine.preparation.converter import UDConverter
from engine.train import train


UDConverter.convert_from_conllu("syntagrus_full.ud", "syntagrus_fixed.txt")
TRAIN_FILENAME = "syntagrus_fixed.txt"


data_model = {
    "char_dropout": 0.2,
    "char_embedding_dim": 24,
    "char_function_hidden_size": 500,
    "char_function_output_size": 200,
    "char_max_word_length": 32,
    "dense_dropout": 0.2,
    "dense_size": 128,
    "gram_dropout": 0.2,
    "gram_hidden_size": 30,
    "rnn_bidirectional": True,
    "rnn_dropout": 0.3,
    "rnn_hidden_size": 128,
    "rnn_input_size": 200,
    "rnn_n_layers": 2,
    "use_chars": True,
    "use_crf": False,
    "use_gram": True,
    "use_trained_char_embeddings": False,
    "use_word_embeddings": False,
    "word_embedding_dropout": 0.2,
    "word_max_count": 10000,
    "use_word_lm": False,
    "use_pos_lm": False
}

with open("model/build_config.json", "w") as write_file:
    json.dump(data_model, write_file)

data_train = {
    "dump_model_freq": 2,
    "epochs_num": 30,
    "external_batch_size": 5000,
    "batch_size": 256,
    "random_seed": 42,
    "rewrite_model": False,
    "sentence_len_groups": [
        [
            26,
            50
        ],
        [
            15,
            25
        ],
        [
            1,
            14
        ]
    ],
    "val_part": 0.05,
    "gram_dict_input": "model/gram_input.json",
    "gram_dict_output": "model/gram_output.json",
    "train_model_config_path": "model/model.json",
    "train_model_weights_path": "model/model.h5",
    "eval_model_config_path": "model/eval_model.json",
    "eval_model_weights_path": "model/eval_model.h5",
    "word_vocabulary": "model/vocabulary.txt",
    "char_set_path": "model/char_set.txt",
    "rewrite_model": True
}

with open("model/train_config.json", "w") as write_file:
    json.dump(data_train, write_file)

f = open('engine/model/acc.txt', 'w')
f.close()

f = open('engine/model/loss.txt', 'w')
f.close()

train(["syntagrus_fixed.txt"],
      train_config_path="model/train_config.json",
      build_config_path="model/build_config.json")




