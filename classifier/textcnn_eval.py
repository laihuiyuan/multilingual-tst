# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
from transformers import MBart50Tokenizer

sys.path.append("")
from classifier.textcnn_train import TextCNN, SCIterator
from classifier.textcnn_train import filter_sizes, num_filters

device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser('Evaluating Style Strength')
    parser.add_argument('-order', default=0, type=str, help='the output order')
    parser.add_argument('-style', default=0, type=int, help='from inf. to for.')
    parser.add_argument('-dim', default=300, type=int, help='the embedding size')
    parser.add_argument('-lang', default='it_IT', type=str, help='language name')
    parser.add_argument('-dataset', default='xformal', type=str, help='dataset name')
    parser.add_argument('-model', default='mbart', type=str, help='the name of model')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random number seed')
    parser.add_argument('-batch_size', default=128, type=int, help='max sents in a batch')
    parser.add_argument("-dropout", default=0.6, type=float, help="Keep prob in dropout")

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    model_path = "facebook/mbart-large-50"
    tokenizer = MBart50Tokenizer.from_pretrained(model_path, src_lang=opt.lang)
    pad_id = tokenizer.pad_token_id

    model = TextCNN(opt.dim, len(tokenizer), filter_sizes, num_filters)
    model_dir = 'checkpoints/textcnn_{}.chkpt'.format(opt.lang)
    model.load_state_dict(torch.load(model_dir))
    model.to(device).eval()
    loss_fn = nn.CrossEntropyLoss()

    test_src, test_tgt = [], []
    if opt.style == 1:
        for i in range(4):
            with open('data/outputs/mbart_{}_{}.1.{}'.format(opt.lang, opt.order, i),'r') as f:
                for line in f.readlines():
                    test_src.append(tokenizer.encode(line.strip()))
    else:
        with open('data/xformal/{}/{}_{}.0'.format(opt.lang, opt.order, opt.lang),'r') as f:
            for line in f.readlines():
                test_tgt.append(tokenizer.encode(line.strip()))
    print('[Info] {} instances from src test set'.format(len(test_src)))
    print('[Info] {} instances from tgt test set'.format(len(test_tgt)))
    test_loader = SCIterator(test_src, test_tgt, pad_id, False, opt)

    corre_num = 0.
    total_num = 0.
    loss_list = []
    with torch.no_grad():
        for batch in test_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            logits = model(src)
            loss = loss_fn(logits, tgt)
            y_hat = logits.argmax(dim=-1)
            same = [int(p == q) for p, q in zip(tgt, y_hat)]
            corre_num += sum(same)
            total_num += len(tgt)
            loss_list.append(loss.item())

    print('[Info] Test: {}'.format('acc {:.2f}% | loss {:.4f}').format(
           corre_num / total_num * 100, np.mean(loss_list)))


if __name__ == '__main__':
    main()
