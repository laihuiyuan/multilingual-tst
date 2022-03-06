# [Multilingual pre-training with Language and Task Adaptation for Multilingual Text Style Transfer (ACL 2022)]

Code coming soon.


## Dependencies
```
python==3.7
pytorch==1.10.0
transformers==4.15.0
```

## Dataset
- [XFORMAL](https://github.com/Elbria/xformal-FoST): informal text (0) <-> formal text (1), e.g. train.0 <-> train.1.
- [News-crawl](http://data.statmt.org/news-crawl/): Language-specific generic non-parallel data.

## Quick Start
### Step 1: Language Adaptation Training
```
python train_lang_adap.py \
       -lang it_IT \
       -dataset news-crawl \
       -batch_size 32 \
       -acc_steps 8 \
       -lr 1e-5 \
       -steps 200000
```

### Step 2: Task Adaptation Training
```
python train_task_adap.py \
       -lang it_IT \
       -dataset xformal \
       -batch_size 32 \
       -acc_steps 1 \
       -lr 1e-5 \
       -style 0
```

### Step 3: Inference

- ADAPT + EN data
```
python infer_en_data.py \
       -lang it_IT \
       -dataset xformal \
       -batch_size 32 \
       -style 0
```

- ADAPT + EN cross-attn
```
python infer_en_attn.py \
       -lang it_IT \
       -dataset xformal \
       -batch_size 32 \
       -style 0
```

## Cite
```
@inproceedings{lai-etal-2022-multi,
    title = "Multilingual pre-training with Language and Task Adaptation for Multilingual Text Style Transfer",
    author = "Lai, Huiyuan  and
      Toral, Antonio  and
      Nissim, Malvina",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    month = May,
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```
