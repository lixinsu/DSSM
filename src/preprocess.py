#!/usr/bin/env python
# coding: utf-8

import json
import jieba
import fire
from tqdm import tqdm

def process_file(infile, outfile, exchange=False):
    fo = open(outfile, 'w')
    cnt = 0
    for line in tqdm(open(infile).readlines()):
        try:
            _, t1, t2, label = line.strip().split('\t')
        except:
            print(line)
            continue
        text1 = [x.strip() for x in jieba.cut(t1) if x.strip()]
        text2 = [x.strip() for x in jieba.cut(t2) if x.strip()]
        text1c = [x for x in t1.strip() if x ]
        text2c = [x for x in t2.strip() if x ]

        ret = {'id': str(cnt), 'text1': text1, 'text2': text2, 'text1c':text1c, 'text2c': text2c, 'label': label}
        fo.write(json.dumps(ret, ensure_ascii=False) + '\n')
        cnt += 1
        ret = {'id': str(cnt), 'text1': text2, 'text2': text1, 'text1c':text2c, 'text2c': text1c, 'label': label}
        fo.write(json.dumps(ret, ensure_ascii=False) + '\n')
        cnt += 1

fire.Fire()
