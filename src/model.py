#!/usr/bin/env python
# coding: utf-8

from gensim.models import Word2Vec
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

from dssm import DSSM


class TMmodel(object):
    """
    """
    def __init__(self, args, word_dict, char_dict):

        self.args = args
        self.word_dict = word_dict
        self.char_dict = char_dict
        self.network = DSSM(args, word_dict, char_dict)
        if args.cuda:
            self.network.cuda()
        self.optimizer = optim.Adamax(self.network.parameters(), weight_decay=0)



    def update(self, ex):
        self.network.train()
        if self.args.cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True))
                                          for e in ex[1:9]]
            label =  Variable(ex[0].cuda(async=True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[1:9]]
            label =  Variable(ex[0])
        scores = self.network(*inputs)
        loss = F.nll_loss(scores, label, weight=torch.Tensor([1, 3.4]).cuda())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.network.parameters(), 10)
        self.optimizer.step()
        return loss.data[0], ex[0].size(0)

    def predict(self, ex):
        self.network.eval()
        if self.args.cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True))
                                          for e in ex[1:9]]
            label =  Variable(ex[0].cuda(async=True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[1:9]]
            label =  Variable(ex[0])
        scores = self.network(*inputs)
        loss = F.nll_loss(scores, label, weight=torch.Tensor([1, 3.4]).cuda())
        scores = torch.exp(scores)
        scores = [x[1] for x in scores.data.tolist()]
        return loss.data[0], ex[0].size(0), scores

    def load_word_embedding(self):
        embedding = self.network.word_embedding.weight.data
        word_dict = self.word_dict
        word_emb = Word2Vec.load(self.args.word_emb_file)
        for w in word_emb.wv.vocab:
            vec = torch.Tensor(word_emb.wv[w])
            embedding[word_dict[w]].copy_(vec)

    def load_char_embedding(self):
        embedding = self.network.char_embedding.weight.data
        char_dict = self.char_dict
        char_emb = Word2Vec.load(self.args.char_emb_file)
        for c in char_emb.wv.vocab:
            vec = torch.Tensor(char_emb.wv[c])
            embedding[char_dict[c]].copy_(vec)

    def save_model(self, model_file):
        params = {
                'args': self.args,
                'word_dict': self.word_dict,
                'char_dict': self.char_dict,
                'state_dict': self.network.state_dict()
                }
        torch.save(params, model_file)


    @staticmethod
    def load_model():
        pass

