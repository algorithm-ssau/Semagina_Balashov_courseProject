from typing import Dict
from engine.prediction import MorphParser
from engine.dop.timer import timeit
from engine.test.estimate import measure


@timeit
def tag(predictor: MorphParser, untagged_filename: str, tagged_filename: str):
    sentences = []
    with open(untagged_filename, "r", encoding='utf-8') as r:
        words = []
        for line in r:
            if line != "\n":
                records = line.strip().split("\t")
                word = records[1]
                words.append(word)
            else:
                sentences.append([word for word in words])
                words = []
    with open(tagged_filename, "w",  encoding='utf-8') as w:
        all_forms = predictor.predict_sentences(sentences)
        for forms in all_forms:
            for i, form in enumerate(forms):
                line = "{}\t{}\t{}\t{}\t{}\n".format(str(i + 1), form.word, form.normal_form, form.pos, form.tag)
                w.write(line)
            w.write("\n")


def tag_files(predictor: MorphParser) -> Dict:
    tag(predictor, "engine/test/test_text.txt", "engine/test/output_text.txt")
    quality = dict()
    print("Test:")
    quality['Test'] = measure("engine/test/gold_text.txt", "engine/test/output_text.txt", True, None)
    count_correct_pos = quality['Test'].correct_pos
    count_tags = quality['Test'].total_tags

    quality['All'] = dict()
    quality['All']['pos_accuracy'] = float(count_correct_pos) / count_tags
    return quality

