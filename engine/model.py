from typing import List, Tuple
import os

import numpy as np
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters
from keras.layers import Input, Embedding, Dense, LSTM, BatchNormalization, Activation, \
    concatenate, Bidirectional, TimeDistributed, Dropout
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras import backend as K

from engine.generator import BatchGenerator
from engine.preparation.gram_vector import GrammemeVectorizer
from engine.preparation.vocab import WordVocabulary
from engine.preparation.loader import Loader
from engine.embeddings import build_dense_chars_layer, get_char_model
from engine.model_object import ConfigModel, ConfigTrain


class ReversedLSTM(LSTM):
    def __init__(self, units, **kwargs):
        kwargs['go_backwards'] = True
        super().__init__(units, **kwargs)

    def call(self, inputs, **kwargs):
        y_rev = super().call(inputs, **kwargs)
        return K.reverse(y_rev, 1)


class LSTMMorphoAnalysis:
    def __init__(self):
        self.language = "ru"
        self.morph = MorphAnalyzer()
        self.converter = converters.converter('opencorpora-int', 'ud14')
        self.grammeme_vectorizer_input = GrammemeVectorizer()
        self.grammeme_vectorizer_output = GrammemeVectorizer()
        self.word_vocabulary = WordVocabulary()
        self.char_set = ""
        self.train_model = None
        self.eval_model = None

    def prepare(self, gram_dump_path_input: str, gram_dump_path_output: str, word_vocabulary_dump_path: str,
                char_set_dump_path: str,
                file_names: List[str] = None) -> None:
        if os.path.exists(gram_dump_path_input):
            self.grammeme_vectorizer_input.load(gram_dump_path_input)
        if os.path.exists(gram_dump_path_output):
            self.grammeme_vectorizer_output.load(gram_dump_path_output)
        if os.path.exists(word_vocabulary_dump_path):
            self.word_vocabulary.load(word_vocabulary_dump_path)
        if os.path.exists(char_set_dump_path):
            with open(char_set_dump_path, 'r', encoding='utf-8') as f:
                self.char_set = f.read().rstrip()
        if self.grammeme_vectorizer_input.is_empty() or \
                self.grammeme_vectorizer_output.is_empty() or \
                self.word_vocabulary.is_empty() or \
                not self.char_set:
            loader = Loader(self.language)
            loader.parse_corpora(file_names)

            self.grammeme_vectorizer_input = loader.grammeme_vectorizer_input
            self.grammeme_vectorizer_input.save(gram_dump_path_input)
            self.grammeme_vectorizer_output = loader.grammeme_vectorizer_output
            self.grammeme_vectorizer_output.save(gram_dump_path_output)
            self.word_vocabulary = loader.word_vocabulary
            self.word_vocabulary.save(word_vocabulary_dump_path)
            self.char_set = loader.char_set
            with open(char_set_dump_path, 'w', encoding='utf-8') as f:
                f.write(self.char_set)

    def save(self, model_config_path: str, model_weights_path: str,
             eval_model_config_path: str, eval_model_weights_path: str):
        if self.eval_model is not None:
            with open(eval_model_config_path, "w", encoding='utf-8') as f:
                f.write(self.eval_model.to_json())
            self.eval_model.save_weights(eval_model_weights_path)
        if self.train_model is not None:
            with open(model_config_path, "w", encoding='utf-8') as f:
                f.write(self.train_model.to_json())
            self.train_model.save_weights(model_weights_path)

    def load_train(self, config: ConfigModel, model_config_path: str = None, model_weights_path: str = None):
        with open(model_config_path, "r", encoding='utf-8') as f:
            custom_objects = {'ReversedLSTM': ReversedLSTM}
            self.train_model = model_from_json(f.read(), custom_objects=custom_objects)
        self.train_model.load_weights(model_weights_path)

        loss = {}
        metrics = {}
        if config.use_crf:
            out_layer_name = 'crf'
            offset = 0
            if config.use_pos_lm:
                offset += 2
            if config.use_word_lm:
                offset += 2
            loss[out_layer_name] = self.train_model.layers[-1 - offset].loss_function
            metrics[out_layer_name] = self.train_model.layers[-1 - offset].accuracy
        else:
            out_layer_name = 'main_pred'
            loss[out_layer_name] = 'sparse_categorical_crossentropy'
            metrics[out_layer_name] = 'accuracy'

        if config.use_pos_lm:
            prev_layer_name = 'shifted_pred_prev'
            next_layer_name = 'shifted_pred_next'
            loss[prev_layer_name] = loss[next_layer_name] = 'sparse_categorical_crossentropy'
            metrics[prev_layer_name] = metrics[next_layer_name] = 'accuracy'
        self.train_model.compile(Adam(clipnorm=5.), loss=loss, metrics=metrics)

        self.eval_model = Model(inputs=self.train_model.inputs, outputs=self.train_model.outputs[0])

    def load_eval(self, config: ConfigModel, eval_model_config_path: str,
                  eval_model_weights_path: str) -> None:
        with open(eval_model_config_path, "r", encoding='utf-8') as f:
            custom_objects = {'ReversedLSTM': ReversedLSTM}
            self.eval_model = model_from_json(f.read(), custom_objects=custom_objects)
        self.eval_model.load_weights(eval_model_weights_path)

    def build(self, config: ConfigModel, word_embeddings=None):
        inputs = []
        embeddings = []

        if config.use_word_embeddings and word_embeddings is not None:
            words = Input(shape=(None,), name='words')
            word_vocabulary_size = word_embeddings.size.shape[0]
            word_embeddings_dim = word_embeddings.size.shape[1]
            words_embedding = Embedding(word_vocabulary_size, word_embeddings_dim, name='word_embeddings')(words)
            embeddings.append(words_embedding)

        if config.use_gram:
            grammemes_input = Input(shape=(None, self.grammeme_vectorizer_input.grammemes_count()), name='grammemes')
            grammemes_embedding = Dropout(config.gram_dropout)(grammemes_input)
            grammemes_embedding = Dense(config.gram_hidden_size, activation='relu')(grammemes_embedding)
            inputs.append(grammemes_input)
            embeddings.append(grammemes_embedding)

        if config.use_chars:
            chars_input = Input(shape=(None, config.char_max_word_length), name='chars')

            char_layer = build_dense_chars_layer(
                max_word_length=config.char_max_word_length,
                char_vocab_size=len(self.char_set) + 1,
                char_emb_dim=config.char_embedding_dim,
                hidden_dim=config.char_function_hidden_size,
                output_dim=config.char_function_output_size,
                dropout=config.char_dropout)
            if config.use_trained_char_embeddings:
                char_layer = get_char_model(
                    char_layer=char_layer,
                    max_word_length=config.char_max_word_length,
                    embeddings=word_embeddings,
                    model_config_path=config.char_model_config_path,
                    model_weights_path=config.char_model_weights_path,
                    vocabulary=self.word_vocabulary,
                    char_set=self.char_set)
            chars_embedding = char_layer(chars_input)
            inputs.append(chars_input)
            embeddings.append(chars_embedding)

        if len(embeddings) > 1:
            layer = concatenate(embeddings, name="LSTM_input")
        else:
            layer = embeddings[0]

        lstm_input = Dense(config.rnn_input_size, activation='relu')(layer)
        lstm_forward_1 = LSTM(config.rnn_hidden_size, dropout=config.rnn_dropout,
                              recurrent_dropout=config.rnn_dropout, return_sequences=True,
                              name='LSTM_1_forward')(lstm_input)

        lstm_backward_1 = ReversedLSTM(config.rnn_hidden_size, dropout=config.rnn_dropout,
                                       recurrent_dropout=config.rnn_dropout, return_sequences=True,
                                       name='LSTM_1_backward')(lstm_input)
        layer = concatenate([lstm_forward_1, lstm_backward_1], name="BiLSTM_input")

        for i in range(config.rnn_n_layers - 1):
            layer = Bidirectional(LSTM(
                config.rnn_hidden_size,
                dropout=config.rnn_dropout,
                recurrent_dropout=config.rnn_dropout,
                return_sequences=True,
                name='LSTM_' + str(i)))(layer)

        layer = TimeDistributed(Dense(config.dense_size))(layer)
        layer = TimeDistributed(Dropout(config.dense_dropout))(layer)
        layer = TimeDistributed(BatchNormalization())(layer)
        layer = TimeDistributed(Activation('relu'))(layer)

        outputs = []
        loss = {}
        metrics = {}
        num_of_classes = self.grammeme_vectorizer_output.size() + 1
        out_layer_name = 'main_pred'
        outputs.append(Dense(num_of_classes, activation='softmax', name=out_layer_name)(layer))
        loss[out_layer_name] = 'sparse_categorical_crossentropy'
        metrics[out_layer_name] = 'accuracy'

        if config.use_pos_lm:
            prev_layer_name = 'shifted_pred_prev'
            next_layer_name = 'shifted_pred_next'
            prev_layer = Dense(num_of_classes, activation='softmax', name=prev_layer_name)
            next_layer = Dense(num_of_classes, activation='softmax', name=next_layer_name)
            outputs.append(prev_layer(Dense(config.dense_size, activation='relu')(lstm_backward_1)))
            outputs.append(next_layer(Dense(config.dense_size, activation='relu')(lstm_forward_1)))
            loss[prev_layer_name] = loss[next_layer_name] = 'sparse_categorical_crossentropy'
            metrics[prev_layer_name] = metrics[next_layer_name] = 'accuracy'

        if config.use_word_lm:
            out_layer_name = 'out_embedding'
            out_embedding = Dense(word_embeddings.shape[0],
                                  weights=[word_embeddings.T, np.zeros(word_embeddings.shape[0])],
                                  activation='softmax', name=out_layer_name, trainable=False)
            outputs.append(out_embedding(Dense(word_embeddings.shape[1], activation='relu')(lstm_backward_1)))
            outputs.append(out_embedding(Dense(word_embeddings.shape[1], activation='relu')(lstm_forward_1)))
            loss[out_layer_name] = 'sparse_categorical_crossentropy'
            metrics[out_layer_name] = 'accuracy'

        self.train_model = Model(inputs=inputs, outputs=outputs)
        self.train_model.compile(Adam(clipnorm=5.), loss=loss, metrics=metrics)
        self.eval_model = Model(inputs=inputs, outputs=outputs[0])
        print(self.train_model.summary())

    def train(self, file_names: List[str], train_config: ConfigTrain, build_config: ConfigModel) -> None:
        np.random.seed(train_config.random_seed)
        sample_counter = self.count_samples(file_names)
        train_idx, val_idx = self.get_split(sample_counter, train_config.val_part)
        for big_epoch in range(train_config.epochs_num):
            print('Main epoch {}'.format(big_epoch))
            batch_generator = BatchGenerator(
                language=self.language,
                file_names=file_names,
                config=train_config,
                grammeme_vectorizer_input=self.grammeme_vectorizer_input,
                grammeme_vectorizer_output=self.grammeme_vectorizer_output,
                build_config=build_config,
                indices=train_idx,
                word_vocabulary=self.word_vocabulary,
                char_set=self.char_set)

            for epoch, (inputs, target) in enumerate(batch_generator):
                self.history = self.train_model.fit(inputs, target, batch_size=train_config.batch_size, epochs=1,
                                                    verbose=2)
                acc = self.history.history['accuracy']
                loss = self.history.history['loss']
                with open("engine/model/acc.txt", "a") as f_acc:
                    f_acc.write(str(np.asarray(acc, dtype=float).mean()) + "\n")
                with open("engine/model/loss.txt", "a") as f_loss:
                    f_loss.write(str(np.asarray(loss, dtype=float).mean()) + "\n")

                if epoch != 0 and epoch % train_config.dump_model_freq == 0:
                    self.save(train_config.train_model_config_path, train_config.train_model_weights_path,
                              train_config.eval_model_config_path, train_config.eval_model_weights_path)
            self.evaluate(
                file_names=file_names,
                val_idx=val_idx,
                train_config=train_config,
                build_config=build_config)

    @staticmethod
    def count_samples(file_names: List[str]):
        sample_counter = 0
        for filename in file_names:
            with open(filename, "r", encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        sample_counter += 1
        return sample_counter

    @staticmethod
    def get_split(sample_counter: int, val_part: float) -> Tuple[np.array, np.array]:
        perm = np.random.permutation(sample_counter)
        border = int(sample_counter * (1 - val_part))
        train_idx = perm[:border]
        val_idx = perm[border:]
        return train_idx, val_idx

    def evaluate(self, file_names, val_idx, train_config: ConfigTrain, build_config: ConfigModel) -> None:
        word_count = 0
        word_errors = 0
        sentence_count = 0
        sentence_errors = 0
        batch_generator = BatchGenerator(
            language=self.language,
            file_names=file_names,
            config=train_config,
            grammeme_vectorizer_input=self.grammeme_vectorizer_input,
            grammeme_vectorizer_output=self.grammeme_vectorizer_output,
            build_config=build_config,
            indices=val_idx,
            word_vocabulary=self.word_vocabulary,
            char_set=self.char_set)
        for epoch, (inputs, target) in enumerate(batch_generator):
            predicted_y = self.eval_model.predict(inputs, batch_size=train_config.batch_size, verbose=0)
            for i, sentence in enumerate(target[0]):
                sentence_has_errors = False
                count_zero = sum([1 for num in sentence if num == [0]])
                real_sentence_tags = sentence[count_zero:]
                answer = []
                for grammeme_probs in predicted_y[i][count_zero:]:
                    num = np.argmax(grammeme_probs)
                    answer.append(num)
                for tag, predicted_tag in zip(real_sentence_tags, answer):
                    tag = tag[0]
                    word_count += 1
                    if tag != predicted_tag:
                        word_errors += 1
                        sentence_has_errors = True
                sentence_count += 1
                if sentence_has_errors:
                    sentence_errors += 1

    def predict_probabilities(self, sentences: List[List[str]], batch_size: int, build_config: ConfigModel) -> List[
        List[List[float]]]:
        max_sentence_len = max([len(sentence) for sentence in sentences])
        if max_sentence_len == 0:
            return [[] for _ in sentences]
        n_samples = len(sentences)

        words = np.zeros((n_samples, max_sentence_len), dtype=np.int)
        grammemes = np.zeros((n_samples, max_sentence_len, self.grammeme_vectorizer_input.grammemes_count()),
                             dtype=np.float)
        chars = np.zeros((n_samples, max_sentence_len, build_config.char_max_word_length), dtype=np.int)

        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
            word_indices, gram_vectors, char_vectors = BatchGenerator.get_sample(
                sentence,
                language=self.language,
                converter=self.converter,
                morph=self.morph,
                grammeme_vectorizer=self.grammeme_vectorizer_input,
                max_word_len=build_config.char_max_word_length,
                word_vocabulary=self.word_vocabulary,
                word_count=build_config.word_max_count,
                char_set=self.char_set)
            words[i, -len(sentence):] = word_indices
            grammemes[i, -len(sentence):] = gram_vectors
            chars[i, -len(sentence):] = char_vectors

        inputs = []
        if build_config.use_word_embeddings:
            inputs.append(words)
        if build_config.use_gram:
            inputs.append(grammemes)
        if build_config.use_chars:
            inputs.append(chars)
        return self.eval_model.predict(inputs, batch_size=batch_size)
