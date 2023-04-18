import datasets
import pandas as pd
import numpy as np
import torch
import os
import transformers
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset,DataLoader
from torch import nn
from torch import cuda
from timeit import default_timer as timer
from tqdm import tqdm, trange
from sklearn import metrics
from importlib.metadata import version
from einops import rearrange 

from discrete_key_value_bottleneck_pytorch import DiscreteKeyValueBottleneck

torch_version = version('torch')
os.environ["CUDA_VISIBLE_DEVICES"]="3"
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
LEARNING_RATE = 3e-5
pool_before = True
##

import torch._dynamo as dynamo

#dynamo.config.verbose = True
#dynamo.config.suppress_errors = True
transformers.logging.set_verbosity_error()

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.premise = dataframe.premise
        self.hypothesis = dataframe.hypothesis
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
      return len(self.premise)

    def __getitem__(self, index):
        premise = str(self.premise[index])
        premise = " ".join(premise.split())
        hypothesis = str(self.hypothesis[index])
        hypothesis = " ".join(hypothesis.split())

        inputs = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True,
            verbose = False
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

#############################################################################
# BERT Model For NLI
#############################################################################

class BERTNLI(torch.nn.Module):
    def __init__(self):
        super(BERTNLI, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.pool_before = pool_before
        
        self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
            encoder = self.encoder,   # pass the frozen encoder into the bottleneck
            dim = 768,                # input dimension
            num_memory_codebooks = 64, # number of heads
            num_memories = 4096,   # number of memories
            dim_memory = 12,        # dimension of the output memories
            decay = 0.9,              # the exponential moving average decay, lower means the keys will change faster
            pool_before = self.pool_before
        )
        self.l2 = torch.nn.Dropout(0.1)
        self.l3 = torch.nn.Linear(768, 3)

    def forward(self, ids, mask,token_type_ids, only_key_optim=False):
        outputs = self.enc_with_bottleneck(ids, mask=mask,token_type_ids=token_type_ids,only_key_optim=only_key_optim)
        #pooled_output = outputs[1]
        #pooled_output = outputs[:,0]        
        if only_key_optim:
           return None
        if not self.pool_before:
            outputs = outputs[:,0]
        dropout_output = self.l2(outputs)
        logits = self.l3(dropout_output)
        if self.pool_before:
           logits = rearrange(logits, 'b 1 h -> b h')
        return logits     


def discrete_key_init(training_loader,model,device):
    model.train()
    optim_iterator = tqdm(training_loader, desc="Iteration")
    for batch_idx, data in enumerate(optim_iterator):
        ids = data['ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        model(ids, mask, token_type_ids, only_key_optim=True)        
    return model
###############################################################################
# Train Model
###############################################################################
def train_model(n_epochs, training_loader, validation1_loader, validation2_loader, model, optimizer, criterion, device):
  
  model.train()
  loss_vals = []
  train_iterator = trange(n_epochs, desc="Epoch")
  for epoch in train_iterator:
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    epoch_iterator = tqdm(training_loader, desc="Iteration")
    train_loss = 0
    valid1_loss = 0
    valid2_loss = 0
    ######################    
    # Train the model #
    ######################
    for batch_idx, data in enumerate(epoch_iterator):
        optimizer.zero_grad()
        #Forward
        ids = data['ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        outputs = model(ids, mask,token_type_ids)
        #Backward
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

    ######################    
    # Validate the model #
    ######################

    model.eval()
    with torch.no_grad():
            # Run validation for matched set
            for batch_idx, data in enumerate(validation1_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                outputs = model(ids, mask, token_type_ids)

                loss = criterion(outputs, targets)
                valid1_loss = valid1_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid1_loss))
            
            # Run validation for mismatched set            
            for batch_idx, data in enumerate(validation2_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                outputs = model(ids, mask, token_type_ids)

                loss = criterion(outputs, targets)
                valid2_loss = valid2_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid2_loss))
                
            # calculate average losses
            train_loss = train_loss / len(training_loader)
            valid1_loss = valid1_loss / len(validation1_loader)
            valid2_loss = valid2_loss / len(validation2_loader)
            # print training/validation statistics
            print('Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Matched Loss: {:.6f}\tAverage Validation Mismatched Loss{:.6f}'.format(
                epoch,
                train_loss,
                valid1_loss,
                valid2_loss
              ))
  return model

def test(testing1_loader,testing2_loader, model, device):
    model.eval()
    fin1_targets=[]
    fin1_outputs=[]
    fin2_targets=[]
    fin2_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing1_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask,token_type_ids)
            preds = outputs.argmax(axis=1)
            fin1_targets.extend(targets.cpu().detach().numpy().tolist())
            fin1_outputs.extend(preds.cpu().detach().numpy().tolist())
        for _, data in enumerate(testing2_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask,token_type_ids)
            preds = outputs.argmax(axis=1)
            fin2_targets.extend(targets.cpu().detach().numpy().tolist())
            fin2_outputs.extend(preds.cpu().detach().numpy().tolist())
    return fin1_outputs, fin1_targets, fin2_outputs, fin2_targets
  
def main():
    
    # Load MNLI
    print("Load MNLI dataset")
    mnli = datasets.load_dataset(path='glue', name='mnli')
    
    # Print MNLI details
    print('The split names in MNLI dataset:')
    for k in mnli:
        print('   ', k)

    print('The number of training examples in mnli dataset:', mnli['train'].num_rows)
    print('The number of validation examples in mnli dataset - part 1:', mnli['validation_matched'].num_rows)
    print('The number of validation examples in mnli dataset - part 2:', mnli['validation_mismatched'].num_rows)
    print('The number of testing examples in mnli dataset -part 1:', mnli['test_matched'].num_rows)
    print('The number of testing examples in mnli dataset -part 2:', mnli['test_mismatched'].num_rows, '\n')

    print('The class names in mnli dataset:', mnli['train'].features['label'].names)
    print('The feature names in mnli dataset:', list(mnli['train'].features.keys()), '\n')

    # Load to DataFrame
    mnli_train_df = pd.DataFrame(mnli['train'])
    mnli_validation_matched_df = pd.DataFrame(mnli['validation_matched'])
    mnli_validation_mismatched_df = pd.DataFrame(mnli['validation_mismatched'])
    mnli_test_matched_df = pd.DataFrame(mnli['test_matched'])
    mnli_test_mismatched_df = pd.DataFrame(mnli['test_mismatched'])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    # Load to Dataset
    train_set =  CustomDataset(mnli_train_df, tokenizer, MAX_LEN)
    matched_val_set = CustomDataset(mnli_validation_matched_df, tokenizer, MAX_LEN)
    mismatched_val_set = CustomDataset(mnli_validation_mismatched_df, tokenizer, MAX_LEN)
    matched_test_set = CustomDataset(mnli_test_matched_df, tokenizer, MAX_LEN)
    mismatched_test_set = CustomDataset(mnli_test_mismatched_df, tokenizer, MAX_LEN)
    #print(tokenizer.decode(training_set[0]['ids']))

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    # Create DataLoaders          
    training_loader = DataLoader(train_set, **train_params)
    matched_validation_loader = DataLoader(matched_val_set, **test_params)
    mismatched_validation_loader = DataLoader(mismatched_val_set, **test_params)
    matched_testing_loader = DataLoader(matched_test_set, **test_params)
    mismatched_testing_loader = DataLoader(mismatched_test_set, **test_params)
    
    model = BERTNLI()
    
    #if int(torch_version[0])>=2:
    #    print("running on new torch")
    #    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    
    optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    
    key_optimizied_model = discrete_key_init(training_loader,model, device)
    print("embed value for first head after optim ",model.enc_with_bottleneck.vq._codebook.embed[0])
    np_array = model.enc_with_bottleneck.vq._codebook.embed[0].detach().cpu().numpy()
    print("np arr ",np_array)
    start = timer() # Start measuring time for Train and Inference
    
    trained_model = train_model(5, training_loader, matched_validation_loader, mismatched_validation_loader, key_optimizied_model, optimizer, criterion, device)
    end = timer() # Stop measuring time for Train
    print("Train time in minutes: ",(end-start)/60)
    
    outputs_matched, targets_matched, outputs_mismatched, targets_mismatched = test(matched_validation_loader,mismatched_validation_loader, trained_model, device)
    
    targets_matched=np.array(targets_matched).astype(int)
    outputs_matched=np.array(outputs_matched).astype(int)
    accuracy_matched = metrics.accuracy_score(targets_matched, outputs_matched)
    f1_score_micro_matched = metrics.f1_score(targets_matched, outputs_matched, average='micro')
    f1_score_macro_matched = metrics.f1_score(targets_matched, outputs_matched, average='macro')
    print("Evaluation of Matched test set")
    print(f"Accuracy Score = {accuracy_matched}")
    print(f"F1 Score (Micro) = {f1_score_micro_matched}")
    print(f"F1 Score (Macro) = {f1_score_macro_matched}")
    
    targets_mismatched=np.array(targets_mismatched).astype(int)
    outputs_mismatched=np.array(outputs_mismatched).astype(int)
    accuracy_mismatched = metrics.accuracy_score(targets_mismatched, outputs_mismatched)
    f1_score_micro_mismatched = metrics.f1_score(targets_mismatched, outputs_mismatched, average='micro')
    f1_score_macro_mismatched = metrics.f1_score(targets_mismatched, outputs_mismatched, average='macro')
    print("Evaluation of Mismatched test set")
    print(f"Accuracy Score = {accuracy_mismatched}")
    print(f"F1 Score (Micro) = {f1_score_micro_mismatched}")
    print(f"F1 Score (Macro) = {f1_score_macro_mismatched}")
    
    end = timer() # Stop measuring time for Train and Inference
    print("Train+inference time in minutes: ",(end-start)/60)

if __name__ == '__main__':
    main()