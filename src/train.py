#!/usr/bin/env python
# coding: utf-8
import os
import argparse

import numpy as np
import torch
from sklearn import metrics


from utils import load_dataset, build_word_dict, build_char_dict, Timer, AverageMeter
from model import TMmodel
from data import MatchDataset,batchify

def init_from_scratch(args, train_exs):
    print('init from scrath')
    print('building word vocabulary')
    word_dict = build_word_dict(args, train_exs)
    print('building char vocabulary')
    char_dict = build_char_dict(args,train_exs)
    model = TMmodel(args, word_dict, char_dict)
    model.load_word_embedding()
    model.load_char_embedding()
    return model


def get_train_args():
    parser = argparse.ArgumentParser()
    runtime = parser.add_argument_group('Environment')
    #runtime.add_argument('--')
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--data-dir', type=str, default='data/atec',
                        help='Directory of training/validation data')
    files.add_argument('--train-file', type=str,
                        default='train_seg.json',
                        help='Preprocessed train file')
    files.add_argument('--dev-file', type=str,
                        default='dev_seg.json',
                        help='Preprocessed dev file')

    files.add_argument('--char-emb-file', type=str,
                        default='data/embedding/w2v_chars.model',
                        help='pretrianed char embedding')
    files.add_argument('--word-emb-file', type=str,
                        default='data/embedding/w2v_words.model',
                        help='pretrain word embedding')
    files.add_argument('--model-dir', type=str, default='data/models')
    files.add_argument('--model-prefix', type=str, default='test')

    general = parser.add_argument_group('General')
    general.add_argument('--cuda', type=bool, default=True)
    general.add_argument('--batch-size', type=int, default=32)
    general.add_argument('--num-epochs', type=int, default=10)
    general.add_argument('--test-batch-size', type=int, default=128)
    general.add_argument('--data-workers', type=int, default=2)
    general.add_argument('--show-steps', type=int, default=100)

    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max-word', type=int, default=10)
    preprocess.add_argument('--max-char', type=int, default=40)
    preprocess.add_argument('--drop-word', type=bool, default=False)
    return parser.parse_args()


def train(args, data_loader, model, stats):
    train_loss = AverageMeter()
    for idx, ex in enumerate(data_loader):
        loss, n =  model.update(ex)
        train_loss.update(loss, n)
        if idx % args.show_steps == 0:
            print('train: Epoch = %d | iter = %d/%d | ' %(  stats['epoch'], idx, len(data_loader) ) +
                              'loss = %.2f | elapsed time = %.2f (s)' % (train_loss.avg, stats['timer'].time()))



def validate(args, data_loader, model, stats):
    dev_loss = AverageMeter()
    scores = []
    gts = []
    for idx, ex in enumerate(data_loader):
        gts.extend(ex[0].tolist())
        loss, n, res = model.predict(ex)
        scores.extend(res)
        dev_loss.update(loss, n)
    truth = np.array(gts, np.int32)
    predictions = np.array(scores)
    print(truth)
    print(predictions)
    f1s = []
    for t in range(18):
        pred = np.array(predictions > t/20.0, np.int32)
        f1 = metrics.f1_score(truth, pred, average='binary')
        print ("F1(%.2f):" % (t/20.0), f1)
        f1s.append(f1)
    print('dev: Epoch = %d | loss = %.2f | f1: %.2f' % (stats['epoch'], dev_loss.avg, max(f1s)) )
    return max(f1s)




def main():
    args = get_train_args()
    train_exs = load_dataset(os.path.join(args.data_dir, args.train_file))
    dev_exs = load_dataset(os.path.join(args.data_dir, args.dev_file))
    model = init_from_scratch(args, train_exs)
    train_dataset = MatchDataset(args, train_exs, model, 'train')
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=batchify,
            pin_memory=args.cuda,
            )
    dev_dataset = MatchDataset(args, dev_exs, model, 'dev')
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=batchify,
            pin_memory=args.cuda,
            )
    stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0}
    best_f1 = 0
    for epoch in range(args.num_epochs):
        stats['epoch'] = epoch
        train(args, train_loader, model, stats)
        f1 = validate(args, dev_loader, model, stats)
        if f1 > best_f1:
            best_f1 = f1
            model.save_model(os.path.join(args.model_dir, '%s-%.4f' % (args.model_prefix, best_f1)))
            stats['best_valid'] = best_f1


if __name__ == "__main__":
    main()
