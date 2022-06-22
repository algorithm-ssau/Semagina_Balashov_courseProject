from typing import List

import numpy as np
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

from engine.model import LSTMMorphoAnalysis
from engine.preparation.tagged import convert_from_opencorpora_tag, process_gram_tag
from engine.preparation.form import WordFormOut
from engine.model_object import ConfigModel


class Predictor:
    def predict(self, words: List[str], include_all_forms: bool) -> List[WordFormOut]:
        raise NotImplementedError()

    def predict_sentences(self, sentences: List[List[str]], batch_size: int,
                          include_all_forms: bool) -> List[List[WordFormOut]]:
        raise NotImplementedError()


class MorphParser(Predictor):
    def __init__(self, eval_model_config_path: str = None, eval_model_weights_path: str = None,
                 gram_dict_input: str = None, gram_dict_output: str = None, word_vocabulary: str = None,
                 char_set_path: str = None, build_config: str = None):

        self.converter = converters.converter('opencorpora-int', 'ud14')
        self.morph = MorphAnalyzer()
        self.build_config = ConfigModel()
        self.build_config.load(build_config)
        self.model = LSTMMorphoAnalysis()
        self.model.prepare(gram_dict_input, gram_dict_output, word_vocabulary, char_set_path)
        self.model.load_eval(self.build_config, eval_model_config_path, eval_model_weights_path)

    def predict(self, words: List[str], include_all_forms: bool = False) -> List[WordFormOut]:
        words_probabilities = self.model.predict_probabilities([words], 1, self.build_config)[0]
        return self.__get_sentence_forms(words, words_probabilities, include_all_forms)

    def predict_sentences(self, sentences: List[List[str]], batch_size: int = 64,
                          include_all_forms: bool = False) -> List[List[WordFormOut]]:
        sentences_probabilities = self.model.predict_probabilities(sentences, batch_size, self.build_config)
        answers = []
        for words, words_probabilities in zip(sentences, sentences_probabilities):
            answers.append(self.__get_sentence_forms(words, words_probabilities, include_all_forms))
        return answers

    def __get_sentence_forms(self, words: List[str], words_probabilities: List[List[float]],
                             include_all_forms: bool) -> List[WordFormOut]:
        result = []
        for word, word_prob in zip(words, words_probabilities[-len(words):]):
            result.append(self.__compose_out_form(word, word_prob[1:], include_all_forms))
        return result

    def __compose_out_form(self, word: str, probabilities: List[float],
                           include_all_forms: bool) -> WordFormOut:
        word_forms = self.morph.parse(word)
        vectorizer = self.model.grammeme_vectorizer_output
        tag_num = int(np.argmax(probabilities))
        score = probabilities[tag_num]
        full_tag = vectorizer.get_name_by_index(tag_num)
        pos, tag = full_tag.split("#")[0], full_tag.split("#")[1]
        lemma = self.__get_lemma(word, pos, tag, word_forms)
        vector = np.array(vectorizer.get_vector(full_tag))
        result_form = WordFormOut(word=word, normal_form=lemma, pos=pos, tag=tag, vector=vector, score=score)

        if include_all_forms:
            weighted_vector = np.zeros_like(vector, dtype='float64')
            for tag_num, prob in enumerate(probabilities):
                full_tag = vectorizer.get_name_by_index(tag_num)
                pos, tag = full_tag.split("#")[0], full_tag.split("#")[1]
                lemma = self.__get_lemma(word, pos, tag, word_forms)
                vector = np.array(vectorizer.get_vector(full_tag), dtype='float64')
                weighted_vector += vector * prob
                form = WordFormOut(word=word, normal_form=lemma, pos=pos, tag=tag, vector=vector, score=prob)
                result_form.possible_forms.append(form)

            result_form.weighted_vector = weighted_vector
        return result_form

    def __get_lemma(self, word: str, pos_tag: str, gram: str, word_forms=None,
                    enable_normalization: bool = True):
        if '_' in word:
            return word

        if word_forms is None:
            word_forms = self.morph.parse(word)
        guess = ""
        max_common_tags = 0
        for word_form in word_forms:
            word_form_pos_tag, word_form_gram = convert_from_opencorpora_tag(self.converter, word_form.tag, word)
            word_form_gram = process_gram_tag(word_form_gram)
            common_tags_len = len(set(word_form_gram.split("|")).intersection(set(gram.split("|"))))
            if common_tags_len > max_common_tags and word_form_pos_tag == pos_tag:
                max_common_tags = common_tags_len
                guess = word_form
        if guess == "":
            guess = word_forms[0]
        if enable_normalization:
            lemma = self.__normalize_for_gikrya(guess)
        else:
            lemma = guess.normal_form
        return lemma


    @staticmethod
    def __normalize_for_gikrya(form):
        if form.tag.POS == 'NPRO':
            if form.normal_form == 'она':
                return 'он'
            if form.normal_form == 'они':
                return 'он'
            if form.normal_form == 'оно':
                return 'он'

        if form.word == 'об':
            return 'об'
        if form.word == 'тот':
            return 'то'
        if form.word == 'со':
            return 'со'

        if form.tag.POS in {'PRTS', 'PRTF'}:
            return form.inflect({'PRTF', 'sing', 'masc', 'nomn'}).word

        return form.normal_form
