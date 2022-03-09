# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from transformers import logging
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

sys.path.append("")
from utils.dataset import collate_fn
from utils.optim import ScheduledOptim

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'


def read_insts(dataset, prefix, lang, tokenizer):

    src_file = 'data/{}/{}_{}.0'.format(dataset, prefix, lang)
    tgt_file = 'data/{}/{}_{}.1'.format(dataset, prefix, lang)

    src_seq, tgt_seq = [], []
    with open(src_file, 'r') as f1, open(tgt_file, 'r') as f2:
        for s, t in zip(f1.readlines(), f2.readlines()):
            s = tokenizer.encode(s[:130])
            t = tokenizer.encode(t[:130])
            src_seq.append(s)
            tgt_seq.append(t)

    return src_seq, tgt_seq


class SCDataset(torch.utils.data.Dataset):
    def __init__(self, insts, label):
        self.insts = insts
        self.label = label

    def __getitem__(self, index):
        return self.insts[index], self.label[index]

    def __len__(self):
        return len(self.insts)


def SCIterator(insts_neg, insts_pos, pad_id, shuffle, opt):
    '''Data iterator for style classifier'''

    def cls_fn(insts):
        x_batch, y_batch = list(zip(*insts))
        x_batch = collate_fn(x_batch, pad_id)
        y_batch = torch.LongTensor(y_batch)

        return (x_batch, y_batch)

    num = len(insts_neg) + len(insts_pos)
    loader = torch.utils.data.DataLoader(
        SCDataset(
            insts=insts_neg + insts_pos,
            label=[0 if i < len(insts_neg)
                   else 1 for i in range(num)]),
        num_workers=2,
        shuffle=shuffle,
        collate_fn=cls_fn,
        batch_size=opt.batch_size)

    return loader


def evaluate(model, valid_loader, epoch, tokenizer):
    '''Evaluation function for style classifier'''
    model.eval()
    corre_num = 0.
    total_num = 0.
    loss_list = []
    with torch.no_grad():
        for batch in valid_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            outs = model(src, mask, labels=tgt)
            loss, logits = outs[:2]
            y_hat = logits.argmax(dim=-1)
            same = [int(p == q) for p, q in zip(tgt, y_hat)]
            corre_num += sum(same)
            total_num += len(tgt)
            loss_list.append(loss.item())
    model.train()
    print('[Info] Epoch {:02d}-valid: {}'.format(
          epoch, 'acc {:.4f} | loss {:.4f}').format(
          corre_num/total_num, np.mean(loss_list)))

    return corre_num/total_num, np.mean(loss_list)


def main():
    parser = argparse.ArgumentParser('The style classifier that is based on mBERT')
    parser.add_argument('-style', default=0, type=int, help='transfer inf. to for.')
    parser.add_argument('-lang', default='en_XX', type=str, help='the name of language')
    parser.add_argument('-lr', default=1e-5, type=float, help='the initial learning rate')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random generator seed')
    parser.add_argument('-log_step', default=100, type=int, help='print log every x steps')
    parser.add_argument('-dataset', default='xformal', type=str, help='the name of dataset')
    parser.add_argument('-eval_step', default=1000, type=int, help='early stopping training')
    parser.add_argument('-batch_size', default=32, type=int, help='maximum sents in a batch')
    parser.add_argument('-epoch', default=50, type=int, help='force stop at specified epoch')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    print('[Info]', opt)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    pad_id = tokenizer.pad_token_id

    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
    model.to(device).train()

    print('[Info] Built a model with {} parameters'.format(
          sum(p.numel() for p in model.parameters())))

    train_src, train_tgt = read_insts(opt.dataset, 'train', opt.lang, tokenizer)
    valid_src, valid_tgt = read_insts(opt.dataset, 'valid', opt.lang, tokenizer)
    print('[Info] {} instances from train set'.format(len(train_src)))
    print('[Info] {} instances from valid set'.format(len(valid_tgt)))
    train_loader = SCIterator(train_src, train_tgt, pad_id, True, opt)
    valid_loader = SCIterator(valid_src, valid_tgt, pad_id, False,opt)

    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                         betas=(0.9, 0.98), eps=1e-09), opt.lr, 1000)

    tab = 0
    avg_acc = 0
    corre_num = 0.
    total_num = 0.
    loss_list = []
    start = time.time()
    for epoch in range(opt.epoch):

        for batch in train_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            outs = model(src, mask, labels=tgt)
            loss, logits = outs[:2]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            y_hat = logits.argmax(dim=-1)
            same = [int(p == q) for p, q in zip(tgt, y_hat)]
            corre_num += sum(same)
            total_num += len(tgt)
            loss_list.append(loss.item())

            if optimizer.cur_step % opt.log_step == 0:
                lr = optimizer._optimizer.param_groups[0]['lr']
                print('[Info] Epoch {:02d}-{:05d}: | acc {:.4f} | '
                      'loss {:.4f} | lr {:.6f} | second {:.2f}'.format(
                      epoch, optimizer.cur_step, corre_num/total_num ,
                      np.mean(loss_list), lr, time.time() - start))
                corre_num = 0.
                total_num = 0.
                loss_list = []
                start = time.time()

            if optimizer.cur_step % opt.eval_step == 0:
                valid_acc, valid_loss = evaluate(model, valid_loader, epoch, tokenizer)
                if avg_acc < valid_acc:
                    avg_acc = valid_acc
                    save_path = 'checkpoints/mbert_sc_{}.chkpt'.format(opt.lang)
                    torch.save(model.state_dict(), save_path)
                    print('[Info] The checkpoint file has been updated.')
                    tab = 0
                else:
                    tab += 1
                    if tab == 5:
                        exit()

if __name__ == '__main__':
    main()
