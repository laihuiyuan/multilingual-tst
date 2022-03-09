# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
from torch.nn.utils import clip_grad_norm_

from transformers import logging
from transformers import MBart50Tokenizer

from utils.optim import ScheduledOptim
from model import shift_tokens_right, MBartWithAdapterForMTST
from utils.dataset import read_insts, BartIterator

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
            decoder_input = shift_tokens_right(
                tgt, tokenizer.pad_token_id, 
                model.config.decoder_start_token_id)
            mask = src.ne(tokenizer.pad_token_id).long()
            outputs = model(src, mask, decoder_input_ids=decoder_input)
            loss = loss_fn(outputs.logits.view(-1, len(tokenizer)), tgt.view(-1))
            loss_list.append(loss.item())
    model.train()
    print('[Info] {:02d}-{:05d} | loss {:.4f}'.format(
          epoch, step, np.mean(loss_list)))

    return np.mean(loss_list)


def main():
    parser = argparse.ArgumentParser('Task adaptation training.')
    parser.add_argument('-lang', default='it_IT', type=str, help='language name')
    parser.add_argument('-style', default=0, type=int, help='transfer inf. to for.')
    parser.add_argument('-lr', default=1e-5, type=float, help='initial earning rate')
    parser.add_argument('-epoch', default=30, type=int, help='force stop at 20 epoch')
    parser.add_argument('-batch_size', default=32, type=int, help='the size in a batch')
    parser.add_argument('-dataset', default='xformal', type=str, help='the dataset name')
    parser.add_argument('-patience', default=3, type=int, help='early stopping fine-tune')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random generator seed')
    parser.add_argument('-log_step', default=100, type=int, help='print logs every x step')
    parser.add_argument('-eval_step', default=1000, type=int, help='evaluate every x step')

    opt = parser.parse_args()
    print('[Info]', opt)
    torch.manual_seed(opt.seed)

    model_name = "facebook/mbart-large-50"
    tokenizer = MBart50Tokenizer.from_pretrained(model_name, src_lang=opt.lang)

    model = MBartWithAdapterForMTST.from_pretrained(model_name)
    checkpoint = 'checkpoints/mbart_lang_adap_{}.chkpt'.format(opt.lang)
    model.load_state_dict(torch.load(checkpoint))

    for param in model.parameters():
        param.requires_grad = False
    for i in range(len(model.model.decoder.layers)):
        for param in model.model.decoder.layers[i].encoder_attn.parameters():
            param.requires_grad = True
    model.to(device).train()

    train_src, train_tgt = read_insts(opt.dataset, opt.style, 'train', 'en_XX')
    valid_src, valid_tgt = read_insts(opt.dataset, opt.style, 'valid', 'en_XX')
    print('[Info] {} instances from train set'.format(len(train_src)))
    print('[Info] {} instances from valid set'.format(len(valid_tgt)))

    train_loader, valid_loader = BartIterator(
        train_src, train_tgt, valid_src, valid_tgt, opt)

    loss_fn =nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                         betas=(0.9, 0.98), eps=1e-09), opt.lr, 1000)

    step = 0
    loss_list = []
    start = time.time()
    tab, eval_loss = 0, 1e8
    for epoch in range(opt.epoch):
        for batch in train_loader: 
            step += 1
            src, tgt = map(lambda x: x.to(device), batch)

            mask = src.ne(tokenizer.pad_token_id).long()
            decoder_input = shift_tokens_right(
                tgt, tokenizer.pad_token_id, model.config.decoder_start_token_id)
            outputs = model(src, mask, decoder_input_ids=decoder_input)
            loss = loss_fn(outputs.logits.view(-1, len(tokenizer)), tgt.view(-1))
            loss_list.append(loss.item())
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
            
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
                valid_loss = evaluate(model, loss_fn, valid_loader, tokenizer, epoch, step)
                if eval_loss >= valid_loss:
                    save_path = 'checkpoints/mbart_en_adap_{}_{}.chkpt'.format(
                        opt.lang, opt.style)
                    torch.save(model.state_dict(), save_path)
                    print('[Info] The checkpoint file has been updated.')
                    eval_loss = valid_loss
                    tab = 0
                else:
                    tab += 1
                if tab == opt.patience:
                    exit()

if __name__ == "__main__":
    main()
