import sys,time
import numpy as np
import torch
import os
import logging
import glob
import math
import json
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import TensorDataset, random_split
import utils
import os
# from apex import amp
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
import torch.autograd as autograd
sys.path.append("./approaches/base/")
from bert_base import Appr as ApprBase
from datasets import load_dataset


class Appr(ApprBase):


    def __init__(self,model,logger, taskcla,args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        return
    
    def discrete_key_init(self,t,data,iter_bar, n_init, is_wiki=False):
        self.model.train()
        key_epoch_runtimes = []
        for i in range (n_init):
            clock0=time.time()
            for step, batch in enumerate(iter_bar):
                if is_wiki:
                    ids = batch['ids'].to(self.device, dtype=torch.long) # For wiki key init 
                    #targets = data['targets'].to(device, dtype=torch.long) # For wiki key init
                    mask = batch['mask'].to(self.device, dtype=torch.long) # For wiki key init 
                    token_type_ids = batch['token_type_ids'].to(self.device, dtype = torch.long) # For wiki key init
                    output_dict = self.model.forward(ids, token_type_ids, mask, True,t)
                else:
                    batch = [bat.to(self.device) if bat is not None else None for bat in batch] # For full and inc key init
                    input_ids, segment_ids, input_mask, targets, _= batch # For full and inc key init
                    output_dict = self.model.forward(input_ids, segment_ids, input_mask, True,t)
                #print("key initting")
            clock1=time.time()
            runtime = clock1 - clock0
            key_epoch_runtimes.append(runtime)
            print('Key Epoch runtime: ', runtime)
        return key_epoch_runtimes
                                
    
    def train(self,t,train,valid,num_train_steps,train_data,valid_data):
        
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
        global_step = 0
        self.model.to(self.device)

        param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad==True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=t_total)
        
        print("learning rate ", self.args.learning_rate)
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        
        key_runtimes = []
        if self.args.key_init == 'inc':
            print(self.args.key_init, " key_init")
            iter_bar = tqdm(train, desc="Key init")
            key_epoch_runtimes = self.discrete_key_init(t,train,iter_bar, 3)          
            key_runtimes.append(key_epoch_runtimes)
            
        epoch_runtimes = []
        
        # Loop epochs
        for e in range(int(self.args.num_train_epochs)):
            # Train
            clock0=time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step)
            clock1=time.time()

            train_loss,train_acc,train_f1_macro=self.eval(t,train)
            clock2=time.time()
            # print('time: ',float((clock1-clock0)*10*25))
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')

            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            
            runtime = clock1 - clock0
            epoch_runtimes.append(runtime)
            print('Epoch runtime: ', runtime)
            
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                print(' *',end='')

            print()
            # break
        # Restore best
        
        # calc avg runtime
        avg_runtime = np.mean(epoch_runtimes)
        print('Average runtime: ', avg_runtime)
        std_runtime = np.std(epoch_runtimes)
        print('Std runtime: ', std_runtime)
        
        utils.set_model_(self.model,best_model)

        return epoch_runtimes, avg_runtime, std_runtime, key_runtimes
    
    def train_epoch(self,t,data,iter_bar,optimizer,t_total,global_step):
        self.model.train()
        #print("model :", self.model)
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, _= batch
            output_dict = self.model.forward(input_ids, segment_ids, input_mask, False, t)
            pooled_rep = output_dict['normalized_pooled_rep']

            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                output=output_dict['y']
                #output = output[t]


            loss=self.ce(output,targets)

            if self.args.sup_loss:
                loss += self.sup_loss(output,pooled_rep,input_ids, segment_ids, input_mask,targets,t)



            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
            loss.backward()

            lr_this_step = self.args.learning_rate * \
                           self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        return global_step

    def eval(self,t,data,test=None,trained_task=None):

        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()
        target_list = []
        pred_list = []

        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, segment_ids, input_mask, targets, _= batch
                real_b=input_ids.size(0)

                output_dict = self.model.forward(input_ids, segment_ids, input_mask, False, t)

                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    output=output_dict['y']
                    #output = output[t]


                loss=self.ce(output,targets)

                _,pred=output.max(1)
                hits=(pred==targets).float()
                target_list.append(targets)
                pred_list.append(pred)
                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=real_b
            f1=self.f1_compute_fn(y_pred=torch.cat(pred_list,0),y_true=torch.cat(target_list,0),average='macro')

        return total_loss/total_num,total_acc/total_num,f1