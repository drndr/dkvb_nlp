# Efficient Continual Learning for Small Language Models with a Discrete Key-Value Bottleneck

This repository contains the code to reproduce our results in our paper 'Efficient Continual Learning for Small Language Models with a Discrete Key-Value Bottleneck'.

Preprint link: https://arxiv.org/abs/2412.08528

The paper proposes a discrete key-value bottleneck (DKVB) for encoder-only NLP models that allows efficient, localized updates for continual learning, reducing catastrophic forgetting and achieving competitive performance with lower computational cost — even in challenging single-head settings without task IDs.

### Folder Structure:
    ├── PyContinual                                # Forked from https://github.com/ZixuanKe/PyContinual Extended with DKVB, includes the main experiments  *note dsc dataset had to be removed due to size
    ├── dkvb                                       # Includes the DKVB model, pre-experiments and single-head CIL experiments
    │   ├── token_seg                              # Includes DKVB variant with token segmentation
    │   ├── datasets_cls                           # Includes datasets used in pre-experiments and single-head CIL experiments *note R52 and 20ng dataset had to be removed due to size

The scripts to reproduce experiments are found in the subfolders
