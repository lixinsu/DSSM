#!/usr/bin/env python
# coding: utf-8

from gensim.models import Word2Vec
from utils import load_dataset

def word_emb(examples):
    sentences = []
    for ex in examples:
        sentences.append(ex['text1'])
        sentences.append(ex['text2'])

    word_dim = 300
    w2v_words_model = Word2Vec(sentences, size=word_dim, window=5, min_count=3, workers=4, iter=5)
    w2v_words_model.save('./data/embedding/w2v_words.model')

    sentences = []
    for ex in examples:
        sentences.append(ex['text1c'])
        sentences.append(ex['text2c'])

    word_dim = 50
    w2v_chars_model = Word2Vec(sentences, size=word_dim, window=5, min_count=3, workers=4, iter=5)
    w2v_chars_model.save('./data/embedding/w2v_chars.model')


if __name__ == "__main__":
    train_exs = load_dataset('data/atec/train_seg.json')
    word_emb(train_exs)
