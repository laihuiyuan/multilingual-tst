# -*- coding: utf-8 -*-

import os

import time
import argparse

import torch
from torch import cuda

from transformers import logging
from transformers import MBart50Tokenizer

from model import MBartWithAdapterForMTST

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nb', default=5, type=int, help='beam search')
    parser.add_argument('-length', default=40, type=int, help='max length')
    parser.add_argument('-bs', default=32, type=int, help='the batch size')
    parser.add_argument("-seed", default=42, type=int, help="the random seed")
    parser.add_argument('-style', default=0, type=int, help='from inf. to for.')
    parser.add_argument('-dataset', default='xformal', type=str, help='dataset')
    parser.add_argument('-lang', default='it_IT', type=str, help='language name')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    model_name = "facebook/mbart-large-50"
    tokenizer = MBart50Tokenizer.from_pretrained(model_name, src_lang=opt.lang)
    bos_token_id=tokenizer.lang_code_to_id[opt.lang]

    model = MBartWithAdapterForMTST.from_pretrained(model_name)
    checkpoint = 'checkpoints/mbart_en_adap_{}_{}.chkpt'.format(
        opt.lang, opt.style)
    model.load_state_dict(torch.load(checkpoint))
    model.to(device).eval()

    src_seq = []
    with open('./data/{}/test_{}.{}'.format(
            opt.dataset, opt.lang, opt.style)) as fin:
        for line in fin.readlines():
            src_seq.append(line.strip())

    with open('./data/outputs/mbart_en_data_{}.{}'.format(
            opt.lang, opt.style), 'w') as fout:
        for idx in range(0, len(src_seq), opt.bs):
            inp = tokenizer.batch_encode_plus(
                src_seq[idx: idx+opt.bs],
                padding=True,
                return_tensors='pt')
            src = inp['input_ids'].to(device)
            mask = inp['attention_mask'].to(device)
            outs = model.generate(
                input_ids=src,
                attention_mask=mask,
                num_beams=opt.nb,
                max_length=opt.length,
                forced_bos_token_id=bos_token_id)
            for x in outs:
                text = tokenizer.decode(
                    x.tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)
                fout.write(text.strip() + '\n')


if __name__ == "__main__":
    main()
