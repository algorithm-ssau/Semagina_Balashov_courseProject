def convert_from_opencorpora_tag(to_ud, tag: str, text: str):
    ud_tag = to_ud(str(tag), text)
    pos = ud_tag.split()[0]
    gram = ud_tag.split()[1]
    return pos, gram


def process_gram_tag(gram: str):
    gram = gram.strip().split("|")
    dropped = ["Animacy", "Aspect", "NumType"]
    gram = [grammem for grammem in gram if sum([drop in grammem for drop in dropped]) == 0]
    return "|".join(sorted(gram)) if gram else "_"
