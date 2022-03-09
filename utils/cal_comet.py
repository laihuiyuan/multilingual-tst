# -*- coding: utf-8 -*-

import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from comet.models import download_model



def cal_comet(file_hyp, file_ref, model):

    scores, srcs = [], []
    with open(file_hyp,'r') as fin:
        cand = []
        for line in fin.readlines():
            srcs.append('')
            cand.append(line.strip())

    for i in range(1):
        refs = []
        with open(file_ref+str(i),'r') as fin:
            for line in fin.readlines():
                refs.append(line.strip())

        data = {"src": srcs, "mt": cand, "ref": refs}
        data = [dict(zip(data, t)) for t in zip(*data.values())]

        scores.extend(model.predict(data, cuda=True, 
                                    show_progress=False)[-1])

    return scores

model = download_model("wmt-large-da-estimator-1719")
scores = cal_comet(sys.argv[1], sys.argv[2], model)
print('The average comet score is {}'.format(np.mean(scores)))
