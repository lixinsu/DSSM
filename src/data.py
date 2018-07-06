#!/usr/bin/env python
# coding: utf-8

import random

import torch

from torch.utils.data import Dataset


class MatchDataset(Dataset):

    def __init__(self, args, examples, model, name):
        self.model = model
        self.examples = examples
        self.name = name

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if self.name == 'train':
            return vectorize(self.examples[index], self.model, is_train=True)
        else:
            return vectorize(self.examples[index], self.model)


def vectorize(ex, model, is_train=False):
    args = model.args
    word_dict = model.word_dict
    char_dict = model.char_dict
    if args.drop_word and is_train:
        if random.random() > 0.5:
            ex['text1'][random.randint(0,len(ex['text1'])-1)] = '<UNK>'
        if random.random() > 0.5:
            ex['text2'][random.randint(0,len(ex['text2'])-1)] = '<UNK>'
    text1 = torch.LongTensor([word_dict.get(w, 1) for w in ex['text1'][:args.max_word]])
    text2 = torch.LongTensor([word_dict.get(w, 1) for w in ex['text2'][:args.max_word]])
    text1c = torch.LongTensor([char_dict.get(w, 1) for w in ex['text1c'][:args.max_char]])
    text2c = torch.LongTensor([char_dict.get(w, 1) for w in ex['text2c'][:args.max_char]])
    label = torch.LongTensor(1).fill_(int(ex['label']))
    return text1, text2, text1c,  text2c, label, ex['id']


def batchify(batch, max_word=10, max_char=40):
    text1s = [ex[0] for ex in batch]
    text2s = [ex[1] for ex in batch]
    text1cs = [ex[2] for ex in batch]
    text2cs = [ex[3] for ex in batch]
    labels = [ex[4] for ex in batch]
    ids = [ex[-1] for ex in batch]

    x1 = torch.LongTensor(len(text1s), max_word).zero_()
    x1_mask = torch.ByteTensor(len(text1s), max_word).fill_(1)
    for i, text in enumerate(text1s):
        x1[i,:text.size(0)].copy_(text)
        x1_mask[i, :text.size(0)].fill_(0)

    x1c = torch.LongTensor(len(text1cs), max_char).zero_()
    x1c_mask =  torch.ByteTensor(len(text1cs), max_char).fill_(1)
    for i, text1c in  enumerate(text1cs):
        x1c[i,:text1c.size(0)].copy_(text1c)
        x1c_mask[i, :text1c.size(0)].fill_(0)

    x2 = torch.LongTensor(len(text2s), max_word).zero_()
    x2_mask = torch.ByteTensor(len(text2s), max_word).fill_(1)
    for i, text2 in enumerate(text2s):
        x2[i, :text2.size(0)].copy_(text2)
        x2_mask[i, :text2.size(0)].fill_(0)

    x2c = torch.LongTensor(len(text2cs), max_char).zero_()
    x2c_mask = torch.ByteTensor(len(text2cs), max_char).fill_(1)
    for i, text2c in enumerate(text2cs):
        x2c[i, :text2c.size(0)].copy_(text2c)
        x2c_mask[i, :text2c.size(0)].fill_(0)

    labels = torch.cat(labels)
    return labels, x1, x2, x1c, x2c,x1_mask, x2_mask, x1c_mask, x2c_mask,  ids



