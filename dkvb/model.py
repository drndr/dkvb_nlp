from transformers import BertModel, RobertaModel, DistilBertModel
from torch import nn, cuda
import torch
from einops import rearrange
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from typing import Tuple
from torchvision import transforms 

from dkv_bn import DiscreteKeyValueBottleneck

class BERTwithBottleNeck(nn.Module):
    def __init__(self, decoder, dim_key, pool_before, pooling_type,n_labels):
        super(BERTwithBottleNeck, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.pool_before = pool_before
        self.pooling = pooling_type
        self.n_labels = n_labels
        
        self.decoder = decoder
        self.dim_key = dim_key
        self.num_key_segments = int(768/dim_key)
        
        if self.decoder == "softmax":
            self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
                encoder = self.encoder,   # pass the frozen encoder into the bottleneck
                decoder = 'values_softmax', # type of decoder: 1 layer mlp or values_softmax(non parametric)
                dim = 768,                # input dimension
                num_key_segments = self.num_key_segments, # number of key segments
                codebook_size = 4096,   # number of different discrete keys in bottleneck codebook
                dim_key = self.dim_key,        # dimension of the key segments
                dim_value = self.n_labels,        # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
                decay = 0.2,              # the exponential moving average decay, lower means the keys will change slower
                pool_before = self.pool_before, # boolean flag whether to pool before or after bottleneck
                pooling_type = self.pooling, # type of pooling : cls or mean
                n_labels = self.n_labels
            )
        else:
            self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
                encoder = self.encoder,   # pass the frozen encoder into the bottleneck
                decoder = 'mlp', # type of decoder: 1 layer mlp or values_softmax(non parametric)
                dim = 768,                # input dimension
                num_key_segments = self.num_key_segments, # number of key segments
                codebook_size = 4096,   # number of different discrete keys in bottleneck codebook
                dim_key = self.dim_key,        # dimension of the key segments
                dim_value = self.dim_key,        # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
                decay = 0.2,              # the exponential moving average decay, lower means the keys will change slower
                pool_before = self.pool_before, # boolean flag whether to pool before or after bottleneck
                pooling_type = self.pooling, # type of pooling : cls or mean
                n_labels = self.n_labels
            )
        
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)

    def forward(self, ids, mask,token_type_ids, key_optim=False):
        outputs = self.enc_with_bottleneck(ids, mask=mask,token_type_ids=token_type_ids,key_optim=key_optim)       
        if key_optim:
           return None # Finish forward pass here during key optimization
        
        #         
        if not self.pool_before:
            if self.pooling == "cls":
               outputs = outputs[:,0] # Pool by CLS token here
            if self.pooling == "mean":
               outputs = outputs.mean(dim=1) # Pool by mean of token dim here
               
        if self.enc_with_bottleneck.decoder=='mlp':
            if self.pool_before:
                outputs = rearrange(outputs, 'b 1 d -> b d')
            dropout_output = self.l2(outputs)
            logits = self.l3(dropout_output)        
        if self.enc_with_bottleneck.decoder=='values_softmax':
            if self.pool_before:
                outputs = rearrange(outputs, 'b 1 h d -> b h d')
            logits = outputs.mean(dim=1)       

        return logits
        
class ROBERTAwithBottleNeck(nn.Module):
    def __init__(self, decoder, dim_key, pool_before, pooling_type,n_labels):
        super(ROBERTAwithBottleNeck, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-base')
        self.pool_before = pool_before
        self.pooling = pooling_type
        self.n_labels = n_labels
        
        self.decoder = decoder
        self.dim_key = dim_key
        self.num_key_segments = int(768/dim_key)
        
        if self.decoder == "softmax":
            self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
                encoder = self.encoder,   # pass the frozen encoder into the bottleneck
                decoder = 'values_softmax', # type of decoder: 1 layer mlp or values_softmax(non parametric)
                dim = 768,                # input dimension
                num_key_segments = self.num_key_segments, # number of key segments
                codebook_size = 4096,   # number of different discrete keys in bottleneck codebook
                dim_key = self.dim_key,        # dimension of the key segments
                dim_value = self.n_labels,        # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
                decay = 0.2,              # the exponential moving average decay, lower means the keys will change slower
                pool_before = self.pool_before, # boolean flag whether to pool before or after bottleneck
                pooling_type = self.pooling, # type of pooling : cls or mean
                n_labels = self.n_labels
            )
        else:
            self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
                encoder = self.encoder,   # pass the frozen encoder into the bottleneck
                decoder = 'mlp', # type of decoder: 1 layer mlp or values_softmax(non parametric)
                dim = 768,                # input dimension
                num_key_segments = self.num_key_segments, # number of key segments
                codebook_size = 4096,   # number of different discrete keys in bottleneck codebook
                dim_key = self.dim_key,        # dimension of the key segments
                dim_value = self.dim_key,        # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
                decay = 0.2,              # the exponential moving average decay, lower means the keys will change slower
                pool_before = self.pool_before, # boolean flag whether to pool before or after bottleneck
                pooling_type = self.pooling, # type of pooling : cls or mean
                n_labels = self.n_labels
            )
        
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)

    def forward(self, ids, mask,token_type_ids, key_optim=False):
        outputs = self.enc_with_bottleneck(ids, mask=mask,token_type_ids=token_type_ids,key_optim=key_optim)       
        if key_optim:
           return None # Finish forward pass here during key optimization
        
        #         
        if not self.pool_before:
            if self.pooling == "cls":
               outputs = outputs[:,0] # Pool by CLS token here
            if self.pooling == "mean":
               outputs = outputs.mean(dim=1) # Pool by mean of token dim here
               
        if self.enc_with_bottleneck.decoder=='mlp':
            if self.pool_before:
                outputs = rearrange(outputs, 'b 1 d -> b d')
            dropout_output = self.l2(outputs)
            logits = self.l3(dropout_output)        
        if self.enc_with_bottleneck.decoder=='values_softmax':
            if self.pool_before:
                outputs = rearrange(outputs, 'b 1 h d -> b h d')
            logits = outputs.mean(dim=1)       

        return logits
        
class DistilBERTwithBottleNeck(nn.Module):
    def __init__(self, decoder, dim_key, pool_before, pooling_type,n_labels):
        super(DistilBERTwithBottleNeck, self).__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pool_before = pool_before
        self.pooling = pooling_type
        self.n_labels = n_labels
        
        self.decoder = decoder
        self.dim_key = dim_key
        self.num_key_segments = int(768/dim_key)
        
        if self.decoder == "softmax":
            self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
                encoder = self.encoder,   # pass the frozen encoder into the bottleneck
                decoder = 'values_softmax', # type of decoder: 1 layer mlp or values_softmax(non parametric)
                dim = 768,                # input dimension
                num_key_segments = self.num_key_segments, # number of key segments
                codebook_size = 4096,   # number of different discrete keys in bottleneck codebook
                dim_key = self.dim_key,        # dimension of the key segments
                dim_value = self.n_labels,        # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
                decay = 0.2,              # the exponential moving average decay, lower means the keys will change slower
                pool_before = self.pool_before, # boolean flag whether to pool before or after bottleneck
                pooling_type = self.pooling, # type of pooling : cls or mean
                n_labels = self.n_labels
            )
        else:
            self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
                encoder = self.encoder,   # pass the frozen encoder into the bottleneck
                decoder = 'mlp', # type of decoder: 1 layer mlp or values_softmax(non parametric)
                dim = 768,                # input dimension
                num_key_segments = self.num_key_segments, # number of key segments
                codebook_size = 4096,   # number of different discrete keys in bottleneck codebook
                dim_key = self.dim_key,        # dimension of the key segments
                dim_value = self.dim_key,        # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
                decay = 0.2,              # the exponential moving average decay, lower means the keys will change slower
                pool_before = self.pool_before, # boolean flag whether to pool before or after bottleneck
                pooling_type = self.pooling, # type of pooling : cls or mean
                n_labels = self.n_labels
            )
        
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)

    def forward(self, ids, mask, key_optim=False):
        outputs = self.enc_with_bottleneck(ids, mask=mask,key_optim=key_optim)       
        if key_optim:
           return None # Finish forward pass here during key optimization
        
        #         
        if not self.pool_before:
            if self.pooling == "cls":
               outputs = outputs[:,0] # Pool by CLS token here
            if self.pooling == "mean":
               outputs = outputs.mean(dim=1) # Pool by mean of token dim here
               
        if self.enc_with_bottleneck.decoder=='mlp':
            if self.pool_before:
                outputs = rearrange(outputs, 'b 1 d -> b d')
            dropout_output = self.l2(outputs)
            logits = self.l3(dropout_output)        
        if self.enc_with_bottleneck.decoder=='values_softmax':
            if self.pool_before:
                outputs = rearrange(outputs, 'b 1 h d -> b h d')
            logits = outputs.mean(dim=1)       

        return logits

class BERTbase(nn.Module):
    def __init__(self,n_labels, is_frozen):
        super(BERTbase, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.n_labels = n_labels
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)
        self.is_frozen = is_frozen
        
        
    def forward(self, ids, mask, token_type_ids, key_optim=False):
        if self.is_frozen:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(ids, attention_mask=mask, token_type_ids=token_type_ids)
                output = outputs[0].mean(dim=1) # mean pooling for frozen BERT
        else:
            outputs = self.encoder(ids, attention_mask=mask, token_type_ids=token_type_ids)
            output = outputs[1] # CLS pooling for full BERT
        dropout_output = self.l2(output)
        logits = self.l3(dropout_output)
        
        return logits
        
class ROBERTAbase(nn.Module):
    def __init__(self,n_labels, is_frozen):
        super(ROBERTAbase, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-base')
        self.n_labels = n_labels
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)
        self.is_frozen = is_frozen
        
        
    def forward(self, ids, mask, token_type_ids, key_optim=False):
        if self.is_frozen:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(ids, attention_mask=mask, token_type_ids=token_type_ids)
                output = outputs[0].mean(dim=1) # mean pooling for frozen RoBERTa
        else:
            outputs = self.encoder(ids, attention_mask=mask, token_type_ids=token_type_ids)
            output = outputs[1] # CLS pooling for full RoBERTa
        dropout_output = self.l2(output)
        logits = self.l3(dropout_output)
        
        return logits
        
class DistilBERTbase(nn.Module):
    def __init__(self,n_labels, is_frozen):
        super(DistilBERTbase, self).__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.n_labels = n_labels
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)
        self.is_frozen = is_frozen
        
        
    def forward(self, ids, mask, key_optim=False):
        if self.is_frozen:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(ids, attention_mask=mask)
                output = outputs[0].mean(dim=1) # mean pooling for frozen DistilBERT
        else:
            outputs = self.encoder(ids, attention_mask=mask)
            output = outputs[0][:,0] # CLS pooling for full DistilBERT
        dropout_output = self.l2(output)
        logits = self.l3(dropout_output)
        
        return logits
        
class BERT_with_EWC(nn.Module):
    # A wrapper class for BERTbase that adds EWC functionality, adapted from https://github.com/ZixuanKe/PyContinual/tree/main/src

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
                
##################################################################
# Implementation of DER++ based on https://github.com/ZixuanKe/PyContinual/tree/main/src

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size: #total batch size
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'segment_ids','input_mask']

    def init_tensors(self,
                     examples: torch.Tensor,
                    segment_ids: torch.Tensor,
                    input_mask: torch.Tensor,
                    labels: torch.Tensor,
                    logits: torch.Tensor,) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))
    def add_data(self, examples, segment_ids=None, input_mask=None, labels=None, logits=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :return:
        """

        #print('segment_ids: ',segment_ids.size())

        if not hasattr(self, 'examples'):
            self.init_tensors(examples, segment_ids,input_mask, labels, logits)

        #print('segment_ids: ',self.segment_ids.size())

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if segment_ids is not None:
                    self.segment_ids[index] = segment_ids[i].to(self.device)
                if logits is not None:
                    self.input_mask[index] = input_mask[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0                
                
class BERT_with_DERpp(nn.Module):
    def __init__(self, n_labels, is_frozen):
        super(BERT_with_DERpp, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.n_labels = n_labels
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)
        self.is_frozen = is_frozen
        
        
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.buffer = Buffer(256, self.device)
        self.mse = torch.nn.MSELoss()
        
        self.alpha = 0.5
        self.beta = 0.5
        
        
    def forward(self, ids, mask, token_type_ids, targets, add_buffer=True):
        if self.is_frozen:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(ids, attention_mask=mask, token_type_ids=token_type_ids)
        else:
            outputs = self.encoder(ids, attention_mask=mask, token_type_ids=token_type_ids)
        dropout_output = self.l2(outputs[1])
        logits = self.l3(dropout_output)
        
        if add_buffer:
            self.buffer.add_data(
                examples=ids,
                segment_ids=token_type_ids,
                input_mask= mask,
                labels=targets,
                #task_labels=torch.ones(samples_per_task,dtype=torch.long).to(self.device) * (t),
                logits = logits
            )
        
        return logits