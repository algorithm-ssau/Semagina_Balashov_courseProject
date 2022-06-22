from engine.genres import tag_files
from engine.prediction import MorphParser
import string
from collections import Counter
import pandas as pd

file = open("engine/test/table/correct.txt", "w")
file.close()
file = open("engine/test/table/incorrect.txt", "w")
file.close()

morph = MorphParser(
    eval_model_config_path="model/eval_model.json",
    eval_model_weights_path="model/eval_model.h5",
    gram_dict_input="model/gram_input.json",
    gram_dict_output="model/gram_output.json",
    word_vocabulary="model/vocabulary.txt",
    char_set_path="model/char_set.txt",
    build_config="model/build_config.json")

quality = tag_files(morph)

dict_morph = {'NUM': 'Числительное', 'PRON': 'Местоимение', 'VERB': 'Глагол', 'ADJ': 'Прилагательное',
              'DET': 'Детерминант', 'NOUN': 'Существительное', 'ADV': 'Наречие '}

c_NUM, c_PRON, c_VERB, c_ADJ, c_DET, c_NOUN, c_ADV = [], [], [], [], [], [], []
i_NUM, i_PRON, i_VERB, i_ADJ, i_DET, i_NOUN, i_ADV = [], [], [], [], [], [], []
j1 = set()
j2 = set()

with open("engine/test/table/correct.txt", "r", encoding="UTF-8") as f:
    for i in f:
        n1 = []
        a = i.split("\n")
        n = a[0].split(":")
        n1.append(n[0])
        j1.add(n[1])
        j2.add(n[2])
        word = morph.predict(n1)
        word1 = word[0].normal_form
        if n[1] == "NUM":
            c_NUM.append(word[0].normal_form + ":" + str(n[2]))
        if n[1] == "PRON":
            c_PRON.append(word[0].normal_form + ":" + str(n[2]))
        if n[1] == "VERB":
            c_VERB.append(word[0].normal_form + ":" + str(n[2]))
        if n[1] == "ADJ":
            c_ADJ.append(word[0].normal_form + ":" + str(n[2]))
        if n[1] == "DET":
            c_DET.append(word[0].normal_form + ":" + str(n[2]))
        if n[1] == "NOUN":
            c_NOUN.append(word[0].normal_form + ":" + str(n[2]))
        if n[1] == "ADV":
            c_ADV.append(word[0].normal_form + ":" + str(n[2]))

print(j1)
print(j2)

with open("engine/test/table/incorrect.txt", "r", encoding="UTF-8") as f:
    for i in f:
        n1 = []
        a = i.split("\n")
        n = a[0].split(":")
        n1.append(n[0])
        word = morph.predict(n1)
        word1 = word[0].normal_form
        if n[1] == "NUM":
            i_NUM.append(word[0].normal_form + ":" + str(n[2]))
        if n[1] == "PRON":
            i_PRON.append(word[0].normal_form + ":" + str(n[2]))
        if n[1] == "VERB":
            i_VERB.append(word[0].normal_form + ":" + str(n[2]))
        if n[1] == "ADJ":
            i_ADJ.append(word[0].normal_form + ":" + str(n[2]))
        if n[1] == "DET":
            i_DET.append(word[0].normal_form + ":" + str(n[2]))
        if n[1] == "NOUN":
            i_NOUN.append(word[0].normal_form + ":" + str(n[2]))
        if n[1] == "ADV":
            i_ADV.append(word[0].normal_form + ":" + str(n[2]))

c_NUM = Counter(c_NUM).most_common()
c_PRON = Counter(c_PRON).most_common()
c_VERB = Counter(c_VERB).most_common()
c_ADJ = Counter(c_ADJ).most_common()
c_DET = Counter(c_DET).most_common()
c_NOUN = Counter(c_NOUN).most_common()
c_ADV = Counter(c_ADV).most_common()

i_NUM = Counter(i_NUM).most_common()
i_PRON = Counter(i_PRON).most_common()
i_VERB = Counter(i_VERB).most_common()
i_ADJ = Counter(i_ADJ).most_common()
i_DET = Counter(i_DET).most_common()
i_NOUN = Counter(i_NOUN).most_common()
i_ADV = Counter(i_ADV).most_common()


def preparation(data):
    n = []
    for i in data:
        n.append(str(i[0]) + ":" + str(i[1]))
    return n


c_NUM = preparation(c_NUM)
c_PRON = preparation(c_PRON)
c_VERB = preparation(c_VERB)
c_ADJ = preparation(c_ADJ)
c_DET = preparation(c_DET)
c_NOUN = preparation(c_NOUN)
c_ADV = preparation(c_ADV)

i_NUM = preparation(i_NUM)
i_PRON = preparation(i_PRON)
i_VERB = preparation(i_VERB)
i_ADJ = preparation(i_ADJ)
i_DET = preparation(i_DET)
i_NOUN = preparation(i_NOUN)
i_ADV = preparation(i_ADV)


def normalize(data, max_len):
    while len(data) < max_len:
        data.append("-")
    return data

file = open("correct_result.xlsx", "w")
file.close()

file = open("incorrect_result.xlsx", "w")
file.close()

def normalize_len(data1, data2, data3, data4, data5, data6, data7):
    len1, len2, len3, len4, len5, len6, len7 = len(data1), len(data2), len(data3), len(data4), len(data5), len(
        data6), len(data7)
    len_all = [len1, len2, len3, len4, len5, len6, len7]
    max_len = max(len_all)
    data1 = normalize(data1, max_len)
    data2 = normalize(data2, max_len)
    data3 = normalize(data3, max_len)
    data4 = normalize(data4, max_len)
    data5 = normalize(data5, max_len)
    data6 = normalize(data6, max_len)
    data7 = normalize(data7, max_len)
    data = {'NUM Числительное': data1, 'PRON Местоимение': data2, 'VERB Глагол': data3, 'ADJ Прилагательное': data4,
            'DET Детерминант': data5, 'NOUN Существительное': data6, 'ADV Наречие': data7}
    df = pd.DataFrame(data)
    return df

df_c = normalize_len(c_NUM, c_PRON, c_VERB, c_ADJ, c_DET, c_NOUN, c_ADV)
df_i = normalize_len(i_NUM, i_PRON, i_VERB, i_ADJ, i_DET, i_NOUN, i_ADV)



df_c.to_excel('correct_result.xlsx', sheet_name='Верная часть речи', index=False)
df_i.to_excel('incorrect_result.xlsx', sheet_name='Неверная часть речи', index=False)

print("Неверное определение части речи")
print(df_i)
print("\n\n")
print("Верное определение части речи")
print(df_c)


