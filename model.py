from transformers import BertModel
from torch import nn
import torch
from einops import rearrange

import numpy as np
from tqdm import tqdm

from dkv_bn import DiscreteKeyValueBottleneck
from copy import deepcopy


class BERTwithBottleNeck(nn.Module):
    def __init__(self, decoder, dim_key, pool_before, pooling_type, n_labels):
        super(BERTwithBottleNeck, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.pool_before = pool_before
        self.pooling = pooling_type
        self.n_labels = n_labels

        self.decoder = decoder
        self.dim_key = dim_key
        self.num_key_segments = int(768 / dim_key)

        if self.decoder == "softmax":
            self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
                encoder=self.encoder,  # pass the frozen encoder into the bottleneck
                decoder='values_softmax',  # type of decoder: 1 layer mlp or values_softmax(non parametric)
                dim=768,  # input dimension
                num_key_segments=self.num_key_segments,  # number of key segments
                codebook_size=4096,  # number of different discrete keys in bottleneck codebook
                dim_key=self.dim_key,  # dimension of the key segments
                dim_value=self.n_labels,
                # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
                decay=0.8,  # the exponential moving average decay, lower means the keys will change slower
                pool_before=self.pool_before,  # boolean flag whether to pool before or after bottleneck
                pooling_type=self.pooling,  # type of pooling : cls or mean
                n_labels=self.n_labels
            )
        else:
            self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
                encoder=self.encoder,  # pass the frozen encoder into the bottleneck
                decoder='mlp',  # type of decoder: 1 layer mlp or values_softmax(non parametric)
                dim=768,  # input dimension
                num_key_segments=self.num_key_segments,  # number of key segments
                codebook_size=4096,  # number of different discrete keys in bottleneck codebook
                dim_key=self.dim_key,  # dimension of the key segments
                dim_value=self.dim_key,
                # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
                decay=0.8,  # the exponential moving average decay, lower means the keys will change slower
                pool_before=self.pool_before,  # boolean flag whether to pool before or after bottleneck
                pooling_type=self.pooling,  # type of pooling : cls or mean
                n_labels=self.n_labels
            )

        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)

    def forward(self, ids, mask, token_type_ids, key_optim=False):
        outputs = self.enc_with_bottleneck(ids, mask=mask, token_type_ids=token_type_ids, key_optim=key_optim)
        if key_optim:
            return None  # Finish forward pass here during key optimization

        #         
        if not self.pool_before:
            if self.pooling == "cls":
                outputs = outputs[:, 0]  # Pool by CLS token here
            if self.pooling == "mean":
                outputs = outputs.mean(dim=1)  # Pool by mean of token dim here

        if self.enc_with_bottleneck.decoder == 'mlp':
            if self.pool_before:
                outputs = rearrange(outputs, 'b 1 d -> b d')
            dropout_output = self.l2(outputs)
            logits = self.l3(dropout_output)
        if self.enc_with_bottleneck.decoder == 'values_softmax':
            if self.pool_before:
                outputs = rearrange(outputs, 'b 1 h d -> b h d')
            logits = outputs.mean(dim=1)

        return logits


class ROBERTAwithBottleNeck(nn.Module):
    def __init__(self, decoder, dim_key, pool_before, pooling_type, n_labels):
        super(ROBERTAwithBottleNeck, self).__init__()
        self.encoder = BertModel.from_pretrained('roberta-base')
        self.pool_before = pool_before
        self.pooling = pooling_type
        self.n_labels = n_labels

        self.decoder = decoder
        self.dim_key = dim_key
        self.num_key_segments = int(768 / dim_key)

        if self.decoder == "softmax":
            self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
                encoder=self.encoder,  # pass the frozen encoder into the bottleneck
                decoder='values_softmax',  # type of decoder: 1 layer mlp or values_softmax(non parametric)
                dim=768,  # input dimension
                num_key_segments=self.num_key_segments,  # number of key segments
                codebook_size=4096,  # number of different discrete keys in bottleneck codebook
                dim_key=self.dim_key,  # dimension of the key segments
                dim_value=self.n_labels,
                # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
                decay=0.8,  # the exponential moving average decay, lower means the keys will change slower
                pool_before=self.pool_before,  # boolean flag whether to pool before or after bottleneck
                pooling_type=self.pooling,  # type of pooling : cls or mean
                n_labels=self.n_labels
            )
        else:
            self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
                encoder=self.encoder,  # pass the frozen encoder into the bottleneck
                decoder='mlp',  # type of decoder: 1 layer mlp or values_softmax(non parametric)
                dim=768,  # input dimension
                num_key_segments=self.num_key_segments,  # number of key segments
                codebook_size=2048,  # number of different discrete keys in bottleneck codebook
                dim_key=self.dim_key,  # dimension of the key segments
                dim_value=self.dim_key,
                # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
                decay=0.8,  # the exponential moving average decay, lower means the keys will change slower
                pool_before=self.pool_before,  # boolean flag whether to pool before or after bottleneck
                pooling_type=self.pooling,  # type of pooling : cls or mean
                n_labels=self.n_labels
            )

        self.dense = nn.Linear(768, 768)
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)

    def forward(self, ids, mask, token_type_ids, key_optim=False):
        outputs = self.enc_with_bottleneck(ids, mask=mask, token_type_ids=token_type_ids, key_optim=key_optim)
        if key_optim:
            return None  # Finish forward pass here during key optimization

        #         
        if not self.pool_before:
            if self.pooling == "cls":
                outputs = outputs[:, 0]  # Pool by CLS token here
            if self.pooling == "mean":
                outputs = outputs.mean(dim=1)  # Pool by mean of token dim here

        if self.enc_with_bottleneck.decoder == 'mlp':
            if self.pool_before:
                outputs = rearrange(outputs, 'b 1 d -> b d')
            # dropout1_output = self.l2(outputs)
            # dense_output = self.dense(dropout1_output)
            # dense_output = torch.tanh(dense_output)
            dropout_output = self.l2(outputs)
            logits = self.l3(dropout_output)
        if self.enc_with_bottleneck.decoder == 'values_softmax':
            if self.pool_before:
                outputs = rearrange(outputs, 'b 1 h d -> b h d')
            logits = outputs.mean(dim=1)

        return logits


class BERTbase(nn.Module):
    def __init__(self, n_labels, is_frozen):
        super(BERTbase, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.n_labels = n_labels
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)
        self.is_frozen = is_frozen

    def forward(self, ids, mask, token_type_ids):
        if self.is_frozen:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(ids, attention_mask=mask, token_type_ids=token_type_ids)
        else:
            outputs = self.encoder(ids, attention_mask=mask, token_type_ids=token_type_ids)
        dropout_output = self.l2(outputs[1])
        logits = self.l3(dropout_output)

        return logits


class BERT_with_EWC(nn.Module):
    # A wrapper class for BERTbase that adds EWC functionality

    def fisher_matrix_diag_bert_dil(self, t, train, device, model, criterion):
        # Init
        fisher = {}
        for n, p in model.named_parameters():
            fisher[n] = 0 * p.data
        # Compute
        model.train()

        for batch in tqdm(train, desc='Fisher diagonal', ncols=100, ascii=True):
            #b = torch.LongTensor(np.arange(i, np.min([i + sbatch, len(train)])))  # .cuda()
            #batch = train[b]
            # batch = [bat.to(device) if bat is not None else None for bat in train]
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch['ids']
            segment_ids = batch['token_type_ids']
            input_mask = batch['mask']
            targets = batch['targets']

            sbatch = input_ids.size(0)

            # Forward and backward
            model.zero_grad()
            output = model.forward(input_ids, input_mask,segment_ids)

            loss = criterion(t, output, targets)
            loss.backward()
            # Get gradients
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += sbatch * p.grad.data.pow(2)
        # Mean
        for n, _ in model.named_parameters():
            fisher[n] = fisher[n] / len(train)
            fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)
        return fisher

    def __init__(self, n_labels, is_frozen):
        super(BERT_with_EWC, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.model = BERTbase(n_labels, is_frozen)
        self.lamb = 5000

    def forward(self, ids, mask, token_type_ids):
        # pass through to the model
        outputs = self.model(ids, mask, token_type_ids)
        return outputs

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return

    def criterion_ewc(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2

        return self.ce(output, targets) + self.lamb * loss_reg

    def ewc(self, t, train_data, device):
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.freeze_model(self.model_old)

        if t > 0:
            fisher_old = {}
            for n, _ in self.model.named_parameters():
                fisher_old[n] = self.fisher[n].clone()
        self.fisher = self.fisher_matrix_diag_bert_dil(t, train_data, device, self.model, self.criterion_ewc)
        if t > 0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n, _ in self.model.named_parameters():
                self.fisher[n] = (self.fisher[n] + fisher_old[n] * t) / (t + 1)
