# -*- coding: utf-8 -*-

import sys
import nltk
from nltk.tokenize import word_tokenize

languages_codes = {
    'en_XX': 'english',
    'pt_XX': 'portuguese',
    'it_IT': 'italian',
    'fr_XX': 'french'
    }
lang = languages_codes[sys.argv[4]]
fin = open(sys.argv[1],'r').readlines()
with open(sys.argv[2],'w') as f:
    for line in fin:
        if sys.argv[3]=='True':
            line = nltk.word_tokenize(line.strip().lower(), language=lang)
        else:
            line = nltk.word_tokenize(line.strip(), language=lang)
        f.write(' '.join(line)+'\n')
