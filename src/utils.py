#!/usr/bin/env python
# coding: utf-8
import time
import json
from collections import defaultdict

def load_dataset(filename):
    examples  = []
    for line in open(filename):
        r = json.loads(line)
        examples.append(r)
    return examples


def build_word_dict(args, examples):
    word2cnt = defaultdict(int)
    for ex in examples:
        for word in ex['text1']:
            word2cnt[word] += 1
        for word in ex['text2']:
            word2cnt[word] += 1
    word_dict = {'<PAD>':0, '<UNK>':1}
    for k,v in word2cnt.items():
        word_dict[k] = len(word_dict)
    return word_dict


def build_char_dict(args, examples):
    word2cnt = defaultdict(int)
    for ex in examples:
        for word in ex['text1c']:
            word2cnt[word] += 1
        for word in ex['text2c']:
            word2cnt[word] += 1
    word_dict = {'<PAD>':0, '<UNK>':1}
    for k,v in word2cnt.items():
        word_dict[k] = len(word_dict)
    return word_dict

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


