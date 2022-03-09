# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
from torch.nn.utils import clip_grad_norm_

from transformers import logging, MBart50Tokenizer

from utils.optim import ScheduledOptim
from model import shift_tokens_right, MBartWithAdapterForMTST
from utils.dataset import (
    word_replace, read_insts_de, BartIterator)


logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'


def evaluate(model, loss_fn, valid_loader, tokenizer, epoch, step):
    '''Evaluation function for mBART'''
    model.eval()
    loss_list = []
    with torch.no_grad():
        for batch in valid_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            src = word_replace(src, mask.sum(-1), 0.3, tokenizer.mask_token_id)

            decoder_input = shift_tokens_right(
                tgt, tokenizer.pad_token_id, 
                model.config.decoder_start_token_id)
            outputs = model(src, mask, decoder_input_ids=decoder_input)
            loss = loss_fn(outputs.logits.view(-1, len(tokenizer)), tgt.view(-1))
            loss_list.append(loss.item())
    model.train()
    print('[Info] {:02d}-{:05d} | loss {:.4f}'.format(
          epoch, step, np.mean(loss_list)))

    return np.mean(loss_list)


def main():
    parser = argparse.ArgumentParser('Language adaptation training')
    parser.add_argument('-lang', default='it_IT', type=str, help='language name')
    parser.add_argument('-acc_steps', default=8, type=int, help='accumulation_steps')
    parser.add_argument('-lr', default=1e-5, type=float, help='initial earning rate')
    parser.add_argument('-steps', default=30, type=int, help='force stop at x steps')
    parser.add_argument('-epoch', default=30, type=int, help='force stop at x epoch')
    parser.add_argument('-dataset', default='news', type=str, help='the dataset name')
    parser.add_argument('-batch_size', default=32, type=int, help='the size in a batch')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random generator seed')
    parser.add_argument('-log_step', default=1000, type=int, help='print log every x step')
    parser.add_argument('-eval_step', default=10000, type=int, help='evaluate every x step')

    opt = parser.parse_args()
    print('[Info]', opt)
    torch.manual_seed(opt.seed)

    model_name = "facebook/mbart-large-50"
    tokenizer = MBart50Tokenizer.from_pretrained(model_name, src_lang=opt.lang)
    model = MBartWithAdapterForMTST.from_pretrained(model_name)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.model.encoder.adapters.parameters():
        param.requires_grad = True
    for param in model.model.decoder.adapters.parameters():
        param.requires_grad = True
    model.to(device).train()
    
    train_src = read_insts_de(opt.dataset, 'train', opt.lang)
    valid_src = read_insts_de(opt.dataset, 'valid', opt.lang)
    train_tgt = train_src.copy()
    valid_tgt = valid_src.copy()
    print('[Info] {} instances from train set'.format(len(train_src)))
    print('[Info] {} instances from valid set'.format(len(valid_tgt)))

    train_loader, valid_loader = BartIterator(
        train_src, train_tgt, valid_src, valid_tgt, opt)

    loss_fn =nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), 
            betas=(0.9, 0.98), eps=1e-09), opt.lr, 1000)

    loss_list = []
    start = time.time()
    step, acc_step = 0, 0
    for epoch in range(opt.epoch):
        for batch in train_loader: 
            step += 1
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            src = word_replace(src, mask.sum(-1), 0.3, tokenizer.mask_token_id)

            decoder_input = shift_tokens_right(
                tgt, tokenizer.pad_token_id, model.config.decoder_start_token_id)
            outputs = model(src, mask, decoder_input_ids=decoder_input)
            loss = loss_fn(outputs.logits.view(-1, len(tokenizer)), tgt.view(-1))

            acc_step += 1
            loss_list.append(loss.item())
            loss = loss / opt.acc_steps
            loss.backward()

            if acc_step == opt.acc_steps:
                clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()
                acc_step = 0
            
            if step % opt.log_step == 0:
                lr = optimizer._optimizer.param_groups[0]['lr']
                print('[Info] {:02d}-{:05d} | loss {:.4f} | '
                      'lr {:.6f} | second {:.1f}'.format(epoch, step,
                      np.mean(loss_list), lr, time.time() - start))
                loss_list = []
                start = time.time()

            if ((len(train_loader) > opt.eval_step
                 and step % opt.eval_step == 0)
                    or (len(train_loader) < opt.eval_step
                        and step % len(train_loader) == 0)):
                evaluate(model, loss_fn, valid_loader, tokenizer, epoch, step)
                save_path = 'checkpoints/mbart_lang_adap_{}.chkpt'.format(opt.lang)
                torch.save(model.state_dict(), save_path)
                print('[Info] The checkpoint file has been updated.')
                if (step >= opt.steps):
                    exit()

if __name__ == "__main__":
    main()
