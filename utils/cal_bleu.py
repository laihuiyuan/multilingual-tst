# -*- coding: utf-8 -*-

import sys
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


hyp, ref = [], []
with open(sys.argv[1],'r') as f:
    for line in f.readlines():
        hyp = nltk.word_tokenize(line.strip(), language=sys.argv[3])

with open(sys.argv[2],'r') as f:
    for line in f.readlines():
        ref.append(nltk.word_tokenize(line.strip(), language=sys.argv[3]))

smooth = SmoothingFunction()
score = sentence_bleu(ref, hyp,smoothing_function=smooth.method1)
print(score)

