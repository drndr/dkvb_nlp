#Coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformers import BertTokenizer as BertTokenizer
import os
import torch
import numpy as np
import random
import nlp_data_utils as data_utils
from nlp_data_utils import ABSATokenizer, InputFeatures
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import math
from datasets import load_dataset

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,mode):

    features = []
    for (ex_index,example) in enumerate(examples):
        labels_a = example['label']
        tokens_a = tokenizer.tokenize(example['text'])

        # print('labels_a: ',labels_a)
        # print('example.text_a: ',example.text_a)
        # print('tokens_a: ',tokens_a)


        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=labels_a))
    return features



glue_tasks = [
    "rte",
    "qqp",
    "sst2",
    "mrpc"
]

def get(logger,args):
    """
    load data for dataset_name
    """
    dataset_name = 'glue'
    classes = glue_tasks

    print('dataset_name: ',dataset_name)

    data={}
    taskcla=[]

    # Others
    f_name = dataset_name + '_random_'+str(args.ntasks)
    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    print('random_sep: ',random_sep)

    dataset = {}
    dataset['train'] = {}
    dataset['valid'] = {}
    dataset['test'] = {}

    task_map = {}
    task_map_inv = {}

    for c_id,cla in enumerate(classes):
        task_map[c_id] = cla
        task_map_inv[cla] = c_id
        d = load_dataset(dataset_name,cla, split='train')
        d_split = d.train_test_split(test_size=0.2,shuffle=True,seed=args.seed)
        dataset['train'][c_id] = d_split['train']

        d_split = d_split['test'].train_test_split(test_size=0.5,shuffle=True,seed=args.seed) # test into half-half

        dataset['test'][c_id] = d_split['test']
        dataset['valid'][c_id] = d_split['train']
        # 50% of test for valid

    examples = {}
    for s in ['train','test','valid']:
        examples[s] = {}
        for c_id, c_data in dataset[s].items():
            nn=c_id #which task_id this class belongs to
            if nn not in examples[s]: examples[s][nn] = []
            for c_dat in c_data:
                # TODO check if correct
                # text= c_dat['text']
                task_name = task_map[c_id]
                if task_name == 'sst2':
                    text = c_dat['sentence']
                elif task_name == 'mrpc':
                    text = c_dat['sentence1'] + ' [SEP] ' + c_dat['sentence2']
                elif task_name == 'qqp':
                    text = c_dat['question1'] + ' [SEP] ' + c_dat['question2']
                elif task_name == 'rte':
                    text = c_dat['sentence1'] + ' [SEP] ' + c_dat['sentence2']
                else:
                    raise ValueError('Unknown task', task_name)
                # label = c_id%class_per_task
                label = c_dat['label']
                examples[s][nn].append(
                    {
                        'text': text,
                        'label': label
                    }
                                       )

    for t in range(len(classes)): # == args.ntasks
        dataset = random_sep[t]
        print('dataset: ',dataset)

        data[t] = {}
        data[t]['name'] = dataset_name + '-' + dataset

        # convert to id
        dataset = task_map_inv[dataset]

        data[t]['ncla'] = 2 # binary classification 0/1

        label_list = [0,1]
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        train_examples = examples['train'][dataset]

        if args.train_data_size > 0:  # TODO: for replicated results, better do outside (in prep_dsc.py), so that can save as a file
            random.Random(args.data_seed).shuffle(train_examples)  # more robust
            train_examples = train_examples[:args.train_data_size]

        num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs
        # num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, "dsc")
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)

        data[t]['train'] = train_data
        data[t]['num_train_steps'] = num_train_steps

        valid_examples = examples['valid'][dataset]
        # No need to change valid for DSC
        # if args.dev_data_size > 0:
        #     random.Random(args.data_seed).shuffle(valid_examples) #more robust
        #     valid_examples = valid_examples[:args.dev_data_size]

        valid_features = convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer, "dsc")
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_all_tasks = torch.tensor([t for f in valid_features], dtype=torch.long)

        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask,
                                   valid_all_label_ids, valid_all_tasks)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        data[t]['valid'] = valid_data


        # tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        eval_examples = examples['test'][dataset]

        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                                    tokenizer, "dsc")

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)
        # Run prediction for full data

        data[t]['test'] = eval_data

        taskcla.append((t, int(data[t]['ncla'])))




    n = 0
    for t in data.keys():
        n += data[t]['ncla']
    data['ncla'] = n


    return data,taskcla