from typing import List
import nltk
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters
from engine.preparation.gram_vector import GrammemeVectorizer
from engine.preparation.vocab import WordVocabulary
from engine.dop.t_open import tqdm_open
from engine.preparation.tagged import convert_from_opencorpora_tag, process_gram_tag


class Loader(object):

    def __init__(self, language: str):
        self.language = language
        self.grammeme_vectorizer_input = GrammemeVectorizer()
        self.grammeme_vectorizer_output = GrammemeVectorizer()
        self.word_vocabulary = WordVocabulary()
        self.char_set = set()
        self.morph = MorphAnalyzer()
        self.converter = converters.converter('opencorpora-int', 'ud14')

    def parse_corpora(self, file_names: List[str]):
        for file_name in file_names:
            with tqdm_open(file_name, encoding="utf-8") as f:
                for line in f:
                    if line == "\n":
                        continue
                    self.__process_line(line)

        self.grammeme_vectorizer_input.init_possible_vectors()
        self.grammeme_vectorizer_output.init_possible_vectors()
        self.word_vocabulary.sort()
        self.char_set = " " + "".join(self.char_set).replace(" ", "")

    def __process_line(self, line: str):
        text, lemma, pos_tag, grammemes = line.strip().split("\t")[0:4]
        self.word_vocabulary.add_word(text.lower())
        self.char_set |= {ch for ch in text}
        self.grammeme_vectorizer_output.add_grammemes(pos_tag, grammemes)
        if self.language == "ru":
            for parse in self.morph.parse(text):
                pos, gram = convert_from_opencorpora_tag(self.converter, parse.tag, text)
                gram = process_gram_tag(gram)
                self.grammeme_vectorizer_input.add_grammemes(pos, gram)
