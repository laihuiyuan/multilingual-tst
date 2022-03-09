# -*- coding: utf-8 -*-

import time
import random

random.seed(1024)
import numpy as np

import torch
import torch.utils.data
from transformers import MBart50Tokenizer
from transformers import MBart50TokenizerFast

model_path = "facebook/mbart-large-50"


def read_insts_de(dataset, prefix, lang):
    tokenizer = MBart50TokenizerFast.from_pretrained(
        model_path, src_lang=lang)
    file = 'data/{}/{}.{}'.format(dataset, prefix, lang)

    seqs = []
    with open(file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            seq_id = tokenizer.encode(line.strip()[:130])
            seqs.append(seq_id)
    del tokenizer

    return seqs


def read_insts(dataset, style, prefix, lang):
    tokenizer = MBart50Tokenizer.from_pretrained(
        model_path, src_lang=lang)

    if style == 0:
        src_file = 'data/{}/{}_{}.0'.format(dataset, prefix, lang)
        tgt_file = 'data/{}/{}_{}.1'.format(dataset, prefix, lang)
    else:
        src_file = 'data/{}/{}_{}.1'.format(dataset, prefix, lang)
        tgt_file = 'data/{}/{}_{}.0'.format(dataset, prefix, lang)

    src_seq, tgt_seq = [], []
    with open(src_file, 'r') as f1, open(tgt_file, 'r') as f2:
        for s, t in zip(f1.readlines(), f2.readlines()):
            s = tokenizer.encode(s[:130])
            t = tokenizer.encode(t[:130])
            src_seq.append(s)
            tgt_seq.append(t)
    del tokenizer

    return src_seq, tgt_seq


def collate_fn(insts, pad_token_id=1):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [pad_token_id] * (max_len - len(inst))
        for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq


def paired_collate_fn(insts):
    src_inst, tgt_inst = list(zip(*insts))
    src_inst = collate_fn(src_inst)
    tgt_inst = collate_fn(tgt_inst)

    return src_inst, tgt_inst


class BartDataset(torch.utils.data.Dataset):
    def __init__(self, src_inst=None, tgt_inst=None):
        self._src_inst = src_inst
        self._tgt_inst = tgt_inst

    def __len__(self):
        return len(self._src_inst)

    def __getitem__(self, idx):
        return self._src_inst[idx], self._tgt_inst[idx]


def BartIterator(train_src, train_tgt, valid_src, valid_tgt, opt):
    '''Data iterator for fine-tuning BART'''

    train_loader = torch.utils.data.DataLoader(
        BartDataset(
            src_inst=train_src,
            tgt_inst=train_tgt),
        num_workers=12,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        BartDataset(
            src_inst=valid_src,
            tgt_inst=valid_tgt),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader


def load_embedding(tokenizer, embed_dim, embed_path=None):
    '''Parse an embedding text file into an array.'''

    embedding = np.random.normal(scale=embed_dim ** -0.5,
                                 size=(len(tokenizer), embed_dim))
    if embed_path == None:
        return embedding

    print('[Info] Loading embedding')
    embed_dict = {}
    with open(embed_path) as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            tokens = line.rstrip().split()
            try:
                embed_dict[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue

    for i in range(len(tokenizer)):
        try:
            word = tokenizer.decode(i)
            if word in embed_dict:
                embedding[i] = embed_dict[word]
        except:
            print(i)

    return embedding


def word_replace(seq, seq_len, replace_prob=0.3, word_idx=3):
    if replace_prob == 0:
        return seq

    noise = torch.rand(seq.size(), dtype=torch.float).to(seq.device)
    pos_idx = torch.arange(seq.size(1)).expand_as(seq).to(seq.device)
    token_mask = (0<pos_idx) & (pos_idx < seq_len.unsqueeze(1)-1)
    drop_mask = (noise < replace_prob) & token_mask
    
    x = seq.clone()
    x.masked_fill_(drop_mask, word_idx)

    return x
