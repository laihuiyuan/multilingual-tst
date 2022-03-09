# -*- coding: utf-8 -*-

import os
import sys
import nltk
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from transformers import logging
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'

def main():
    parser = argparse.ArgumentParser('Evaluating Style Strength')
    parser.add_argument('-order', default=0, type=str, help='the training order')
    parser.add_argument('-style', default=0, type=int, help='from informal to formal')
    parser.add_argument('-lang', default='en_XX', type=str, help='the name of language')
    parser.add_argument('-model_dir', default='checkpoint/', type=str, help='model_dir')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random number seed')
    parser.add_argument('-batch_size', default=256, type=int, help='max sent. in a batch')
    parser.add_argument('-dataset', default='xformal', type=str, help='the dataset name')

    opt = parser.parse_args()
    opt.model_dir = 'checkpoints/xlm-r/'

    torch.manual_seed(opt.seed)
    config = AutoConfig.from_pretrained(
        opt.model_dir,
        num_labels=1,
        finetuning_task=None,
        cache_dir=None,
        revision='main',
        use_auth_token=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        opt.model_dir,
        cache_dir=None,
        use_fast=True,
        revision='main',
        use_auth_token=True ,
    )


    model = AutoModelForSequenceClassification.from_pretrained(
        opt.model_dir,
        config=config,
        cache_dir=None,
        revision='main',
        use_auth_token=True,
    )
    model.to(device)


    test = []
    if opt.style == 1:
        for i in range(4):
            with open('data/outputs/mbart_{}_{}.1.{}'.format(opt.lang, opt.order, i),'r') as f:
                for line in f.readlines():
                   test.append(' '.join(nltk.word_tokenize(line.strip())))
    else:
        with open('data/outputs/mbart_{}_{}.0'.format(opt.lang, opt.order),'r') as f:
            for line in f.readlines():
                test.append(' '.join(nltk.word_tokenize(line.strip())))
    print('[Info] {} instances from test set'.format(len(test)))

    scores = []
    with torch.no_grad():
        for idx in range(0, len(test), opt.batch_size):
            inp = tokenizer.batch_encode_plus(
                        test[idx: idx+opt.batch_size],
                        padding=True, return_tensors='pt')
            src = inp['input_ids'].to(device)
            mask = inp['attention_mask'].to(device)
            outs = model(src, mask)
            scores.extend(outs.logits.squeeze(1).cpu().tolist())

    print('[Info] Test: {}'.format('avg score {:.4f}').format(
           np.mean(scores)))

if __name__ == '__main__':
    main()
