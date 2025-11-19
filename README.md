## Source Code for CONTINUAL LEARNING WITH A DISCRETE KEY-VALUE BOTTLENECK IN PRE-TRAINED LANGUAGE MODELS

### Folder Structure:
    ├── PyContinual                                # Forked from https://github.com/ZixuanKe/PyContinual Extended with DKVB, includes the main experiments  *note dsc dataset had to be removed due to size
    ├── dkvb                                       # Includes the DKVB model, pre-experiments and single-head CIL experiments
    │   ├── token_seg                              # Includes DKVB variant with token segmentation
    │   ├── datasets_cls                           # Includes datasets used in pre-experiments and single-head CIL experiments *note R52 and 20ng dataset had to be removed due to size

The scripts to reproduce experiments are found in the subfolders
