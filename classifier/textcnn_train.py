# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from transformers import MBart50Tokenizer
from transformers import MBartForConditionalGeneration

sys.path.append("")
from utils.dataset import collate_fn
from utils.dataset import read_insts
from utils.optim import ScheduledOptim

filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'


def load_embedding(tokenizer, embed_dim, embed_path=None):
    '''Parse an embedding text file into an torch.nn.Embedding layer.'''

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


def evaluate(model, valid_loader, loss_fn, epoch):
    '''Evaluation function for style classifier'''
    model.eval()
    total_acc = 0.
    total_num = 0.
    total_loss = []
    with torch.no_grad():
        for batch in valid_loader:
            x_batch, y_batch = map(lambda x: x.to(device), batch)
            logits = model(x_batch)
            total_loss.append(loss_fn(logits, y_batch).item())
            _, y_hat = torch.max(logits,dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            total_acc += sum(same)
            total_num += len(y_batch)
    model.train()
    print('[Info] Epoch {:02d}-valid | '
          'acc {:.2f}% | loss {:.4f}'.format(epoch,
          total_acc/total_num*100, np.mean(total_loss)))

    return total_acc/total_num, np.mean(total_loss)

    
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
        src, tgt = list(zip(*insts))
        src = collate_fn(src, pad_id)
        tgt = torch.LongTensor(tgt)

        return (src, tgt)

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


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, embeding):
        super(EmbeddingLayer, self).__init__()
        self.embeding = nn.Embedding(vocab_size, embed_dim)
        if embeding is not None:
            self.embeding.weight.data = embeding

    def forward(self, x):
        if len(x.size()) == 2:
            y = self.embeding(x)
        else:
            y = torch.matmul(x, self.embeding.weight)
        return y


class TextCNN(nn.Module):
    '''A TextCNN Classification Model'''

    def __init__(self, embed_dim, vocab_size, filter_sizes, 
                 num_filters, embedding=None, dropout=0.0):
        super(TextCNN, self).__init__()

        self.feature_dim = sum(num_filters)
        self.embeder = EmbeddingLayer(vocab_size, embed_dim, embedding)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim))
            for (n, f) in zip(num_filters, filter_sizes)
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            self.dropout,
            nn.Linear(self.feature_dim, int(self.feature_dim / 2)), nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), 2)
        )

    def forward(self, inp):
        inp = self.embeder(inp).unsqueeze(1)
        convs = [F.relu(conv(inp)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        out = torch.cat(pools, 1)
        logit = self.fc(out)

        return logit

    def build_embeder(self, vocab_size, embed_dim, embedding=None):
        embeder = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(embeder.weight, mean=0, std=embed_dim ** -0.5)
        if embedding is not None:
            embeder.weight.data = torch.FloatTensor(embedding)

        return embeder


def main():
    parser = argparse.ArgumentParser('TextCNN Style Classifier')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-dim', default=300, type=int, help='the embedding size')
    parser.add_argument('-style', default=0, type=int, help='transfer inf. to for.')
    parser.add_argument('-dataset', default='xformal', type=str, help='dataset name')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random number seed')
    parser.add_argument('-lang', default='it_IT', type=str, help='the name of language')
    parser.add_argument('-dropout', default=.5, type=float, help="Keep prob in dropout.")
    parser.add_argument('-log_step', default=100, type=int, help='print log every x steps')
    parser.add_argument('-epoch', default=10, type=int, help='force stop at specified epoch')
    parser.add_argument('-eval_step', default=1000, type=int, help='early stopping training')
    parser.add_argument('-batch_size', default=24, type=int, help='maximum sents in a batch')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    print('[Info]', opt)

    model_path = "facebook/mbart-large-50"
    tokenizer = MBart50Tokenizer.from_pretrained(model_path, src_lang=opt.lang)
    pad_id = tokenizer.pad_token_id
    
    #if os.path.exists('checkpoints/embedding.pt'):
    #    embedding = torch.load('checkpoints/embedding.pt')
    #else:
    #    embed_path = '../../data/glove.840B.300d.txt'
    #    embedding = load_embedding(tokenizer, 300, embed_path)
    #    torch.save(embedding, 'checkpoints/embedding.pt')

    model = TextCNN(opt.dim, len(tokenizer), filter_sizes, 
                    num_filters, None, opt.dropout)
    model.to(device).train()
    print('[Info] Built a model with {} parameters'.format(
           sum(p.numel() for p in model.parameters())))

    train_src, train_tgt, valid_src, valid_tgt = [], [], [], []
    for lang in [opt.lang]:
        train_src_lang, train_tgt_lang = read_insts(opt.dataset, opt.style, 'train', lang)
        valid_src_lang, valid_tgt_lang = read_insts(opt.dataset, opt.style, 'valid', lang)
        train_src = train_src + train_src_lang
        train_tgt = train_tgt + train_tgt_lang
        valid_src = valid_src + valid_src_lang
        valid_tgt = valid_tgt + valid_tgt_lang
    print('[Info] {} instances from train set'.format(len(train_src)))
    print('[Info] {} instances from valid set'.format(len(valid_tgt)))
    train_loader = SCIterator(train_src, train_tgt, pad_id, True, opt)
    valid_loader = SCIterator(valid_src, valid_tgt, pad_id, False,opt)

    loss_fn = nn.CrossEntropyLoss()
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
            optimizer.zero_grad()
            outs = model(src)
            loss = loss_fn(outs, tgt)
            loss.backward()
            optimizer.step()

            y_hat = outs.argmax(dim=-1)
            same = [int(p == q) for p, q in zip(tgt, y_hat)]
            corre_num += sum(same)
            total_num += len(tgt)
            loss_list.append(loss.item())

            if optimizer.cur_step % opt.log_step == 0:
                lr = optimizer._optimizer.param_groups[0]['lr']
                print('[Info] Epoch {:02d}-{:05d} | acc {:.2f}% | '
                    'loss {:.4f} | lr {:.5f} | second {:.2f}'.format(
                    epoch, optimizer.cur_step, corre_num / total_num * 100,
                    np.mean(loss_list), lr, time.time() - start))
                corre_num = 0.
                total_num = 0.
                loss_list = []
                start = time.time()

            if optimizer.cur_step % opt.eval_step == 0:
                valid_acc, valid_loss = evaluate(model, valid_loader, loss_fn, epoch)
                if avg_acc < valid_acc:
                    avg_acc = valid_acc
                    save_path = 'checkpoints/textcnn_{}.chkpt'.format(opt.lang)
                    torch.save(model.state_dict(), save_path)
                    print('[Info] The checkpoint file has been updated.')
                    tab = 0
                else:
                    tab += 1
                    if tab == 5:
                        exit()


if __name__ == '__main__':
    main()



