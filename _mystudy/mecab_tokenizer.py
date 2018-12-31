from konlpy.tag import Mecab

mecab = Mecab()

def mecab_tokenizer(text):
    return mecab.morphs(text)
