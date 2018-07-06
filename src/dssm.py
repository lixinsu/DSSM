#!/usr/bin/env python
# coding: utf-8

import ipdb
import torch
import torch.nn as nn
from torch.nn import functional as F
import layers

class DSSM(nn.Module):
    def __init__(self, args, word_dict, char_dict):
        super(DSSM, self).__init__()
        args.word_dim = 300
        args.char_dim = 50
        args.word_hidden = 256
        args.char_hidden = 128
        self.args  = args

        # word layers
        self.word_embedding = nn.Embedding(len(word_dict), args.word_dim, padding_idx=0)
        self.word_gru_bi = nn.GRU(args.word_dim, args.word_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.word_gru1 = nn.GRU(args.word_dim, args.word_hidden, num_layers=2, batch_first=True)
        self.self_word_attn = layers.LinearSeqAttn(args.word_dim)

        # char layers
        self.char_embedding = nn.Embedding(len(char_dict), args.char_dim, padding_idx=0)
        self.self_char_attn = layers.LinearSeqAttn(args.char_dim)
        self.char_gru = nn.GRU(args.char_dim, args.char_hidden, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear( 6 * args.word_hidden + 2 * args.word_dim + args.char_hidden * 4 + 2 * args.char_dim, 32)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear( 6 * args.word_hidden + 2 * args.word_dim + args.char_hidden * 4 + 2 * args.char_dim , 48)
        self.act2 = nn.Sigmoid()
        self.linear3 = nn.Linear(80, 2)
        self.act3 = nn.LogSoftmax()
        self.dropout = nn.Dropout(0.5)


    def forward(self, x1, x2, x1c, x2c, x1_mask, x2_mask, x1c_mask, x2c_mask):
        """
        :param x1: b*max_word text 1 word ids
        :param x2: b*max_word text 2 word ids
        :param x1c: b*max_char text 1 char ids
        :param x2c: b*max_char text 2 char ids
        """
        x1_emb = self.word_embedding(x1)
        x2_emb = self.word_embedding(x2)
        x1c_emb = self.char_embedding(x1c)
        x2c_emb = self.char_embedding(x2c)
        #  word operation

        _, x1_enc_bi = self.word_gru_bi(x1_emb)
        x1_enc_bi = torch.cat(list(x1_enc_bi), dim=1)

        _, x2_enc_bi = self.word_gru_bi(x2_emb)
        x2_enc_bi = torch.cat(list(x2_enc_bi), dim=1)

        _, x1_enc2 = self.word_gru1(x1_emb)
        x1_enc2 =  torch.cat(list(x1_enc2), dim=1)

        _, x2_enc2 = self.word_gru1(x2_emb)
        x2_enc2 =  torch.cat(list(x2_enc2), dim=1)

        x1_enc_att = self.self_word_attn(x1_emb, x1_mask)
        x2_enc_att = self.self_word_attn(x2_emb, x2_mask)

        v1 = torch.cat([x1_enc_att, x1_enc_bi], dim=1)
        v2 = torch.cat([x2_enc_att, x2_enc_bi], dim=1)

        # char operation
        _, x1c_enc_bi = self.char_gru(x1c_emb)
        x1c_enc_bi = torch.cat(list(x1c_enc_bi), dim=1)
        _, x2c_enc_bi = self.char_gru(x2c_emb)
        x2c_enc_bi = torch.cat(list(x2c_enc_bi), dim=1)

        x1c_enc_att = self.self_char_attn(x1c_emb, x1c_mask)
        x2c_enc_att = self.self_char_attn(x2c_emb, x2c_mask)

        x1c_enc = torch.cat([x1c_enc_att, x1c_enc_bi], dim=1)
        x2c_enc = torch.cat([x2c_enc_att, x2c_enc_bi], dim=1)

        mul = v1 * v2
        sub = torch.abs(v1 - v2)
        #maxi = torch.max([v1*v1, v2*v2])

        mulc = x1c_enc * x2c_enc
        subc = torch.abs(x1c_enc - x2c_enc)

        sub2 = torch.abs(x1_enc2 - x2_enc2)
        #
        matchlist = torch.cat([mul, sub, mulc, subc, sub2], dim=1)

        matchlist = self.dropout(matchlist)

        matchlist = torch.cat([ self.act1(self.linear1(matchlist)), self.act2(self.linear2(matchlist))], dim=1)


        out = self.act3(self.linear3(matchlist))

        return out







