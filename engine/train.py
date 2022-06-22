import os
from typing import List
from engine.model import LSTMMorphoAnalysis
from engine.model_object import ConfigModel, ConfigTrain
from engine.dop.embedding import load_embeddings


def train(file_names: List[str], train_config_path: str, build_config_path: str, embeddings_path: str = None):
    train_config = ConfigTrain()
    train_config.load(train_config_path)
    build_config = ConfigModel()
    build_config.load(build_config_path)
    model = LSTMMorphoAnalysis()
    model.prepare(train_config.gram_dict_input, train_config.gram_dict_output,
                  train_config.word_vocabulary, train_config.char_set_path, file_names)

    if os.path.exists(train_config.eval_model_config_path) and not train_config.rewrite_model:
        model.load_train(build_config, train_config.train_model_config_path, train_config.train_model_weights_path)
        print(model.eval_model.summary())
    else:
        embeddings = None
        if embeddings_path is not None:
            embeddings = load_embeddings(embeddings_path, model.word_vocabulary, build_config.word_max_count)
        model.build(build_config, embeddings)
    # keras.utils.plot_model(model.eval_model, "../model.png")
    model.train(file_names, train_config, build_config)
