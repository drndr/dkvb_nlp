import numpy as np
import torch
import os
import transformers
from torch.utils.data import DataLoader, RandomSampler
from torch import cuda, nn
from timeit import default_timer as timer
from tqdm import tqdm, trange
from sklearn import metrics
import wandb

from dkv_bn import DiscreteKeyValueBottleneck
from utils import load_glue_dataset, load_cls_dataset, load_class_increment_20ng, load_class_increment_r8
from model import BERTwithBottleNeck, BERTbase

import argparse

transformers.logging.set_verbosity_error()

os.environ["CUDA_VISIBLE_DEVICES"]="0"

###############################################################################
# Optimize Keys without backward pass
###############################################################################
def discrete_key_init(n_epochs,training_loader,model,device):
    train_iterator = trange(n_epochs, desc="Epoch")
    for epoch in train_iterator:
        optim_iterator = tqdm(training_loader, desc="Iteration")
        for batch_idx, data in enumerate(optim_iterator):
            ids = data['ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            model(ids, mask, token_type_ids, key_optim=True)        
    return model
###############################################################################
# Train Model
###############################################################################
def train_model(start_epoch, n_epochs, training_loader, validation1_loader, model, optimizer, criterion, device, wandb_enabled):
  
  model.train()
  loss_vals = []
  train_iterator = trange(start_epoch,n_epochs+1, desc="Epoch")
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
        outputs = model(ids, mask, token_type_ids)
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
                
    # calculate average losses
    train_loss = train_loss / len(training_loader)
    valid1_loss = valid1_loss / len(validation1_loader)
    # print training/validation statistics
    print('Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
        epoch,
        train_loss,
        valid1_loss
    ))
    if wandb_enabled:
        wandb.log({"Epoch": epoch,
                   "Train loss": train_loss,
                   "Valid loss":valid1_loss
                 })
  return model

def test(testing1_loader, model, device):
    model.eval()
    fin1_targets=[]
    fin1_outputs=[]
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
    return fin1_outputs, fin1_targets
  
def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("epochs", default=1, type=int, help="Number of training epochs")
    parser.add_argument("batch_size", default=16, type=int, help="Batch size for training and testing")
    parser.add_argument("lr_global", default=3e-5, type=float, help="Learning rate for decoder")
    parser.add_argument("lr_values", default=3e-2, type=float, help="Learning rate for values in bottleneck")
    parser.add_argument("pooling", default="cls", type=str, help="Type of poolings (cls, mean)")
    parser.add_argument("--wandb_enabled", action="store_true", help="wandb monitoring")

    args = parser.parse_args()

    if args.wandb_enabled:
        wandb.init(project="R8 increment", entity="drndr21", name="bert bottleneck softmax")
        #wandb.init(project="R8 increment", entity="drndr21", name=str(args.epochs)+"e "+str(args.batch_size)+"b "+str(args.lr_global)+"Global "+str(args.lr_values)+"Values "+args.pooling+"Ptype ")
    
    # Load to Dataset
    class_sets = load_class_increment_20ng()
    full_train, val_set, n_classes = load_cls_dataset("20ng", max_len=512)
    
        
    # Create validation set here (is reused for all class increment testing)
    validation_params = {'batch_size': args.batch_size,
                'shuffle': True
                }             
    validation_loader = DataLoader(val_set, **validation_params)
    
    model = BERTwithBottleNeck("softmax", 12, False, args.pooling, n_classes)
    #model = BERTbase(n_classes, False)
    
    optimizer = torch.optim.AdamW([{"params": model.enc_with_bottleneck.parameters(), "lr":args.lr_values},
                                  {"params": model.l3.parameters()}],
                                  lr=args.lr_global)
            
    criterion = nn.CrossEntropyLoss()
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)    
    
    start = timer() # Start measuring time for Key Init, Train and Inference
    
    
    train_params = {'batch_size': args.batch_size,
                    'shuffle': False
                    }   
    training_loader = DataLoader(full_train, **train_params)
    trained_model = discrete_key_init(3,training_loader, model, device)
    #trained_model = model
    
    
    end = timer() # Stop measuring time for Key Init
    print("Key Init time in minutes: ",(end-start)/60)
    
    if args.wandb_enabled:
        wandb.watch(model)
    
    # Class incremental training and testing loop
    start_epoch = 1
    end_epoch = args.epochs+1
    for i in range(len(class_sets)):
    
        train_set = class_sets[i]
        sampler = RandomSampler(train_set, replacement=True, num_samples=100)
        
        train_params = {'sampler':sampler,
                        'batch_size': args.batch_size,
                        'shuffle': False
                       }   
        training_loader = DataLoader(train_set, **train_params)        
        #trained_model = discrete_key_init(1,training_loader, trained_model, device)
        trained_model = train_model(start_epoch, end_epoch, training_loader, validation_loader, trained_model, optimizer, criterion, device, args.wandb_enabled)
        start_epoch = start_epoch+args.epochs
        end_epoch = end_epoch+args.epochs
    
        outputs_matched, targets_matched = test(validation_loader, trained_model, device)
    
        targets_matched=np.array(targets_matched).astype(int)
        outputs_matched=np.array(outputs_matched).astype(int)
        accuracy_matched = metrics.accuracy_score(targets_matched, outputs_matched)
        f1_score_micro_matched = metrics.f1_score(targets_matched, outputs_matched, average='micro')
        f1_score_macro_matched = metrics.f1_score(targets_matched, outputs_matched, average='macro')
        print("Evaluation of test set")
        print(f"Accuracy Score = {accuracy_matched}")
        print(f"F1 Score (Micro) = {f1_score_micro_matched}")
        print(f"F1 Score (Macro) = {f1_score_macro_matched}")
    
    end = timer() # Stop measuring time for Train and Inference
    print("Key Init + Train + inference time in minutes: ",(end-start)/60)

if __name__ == '__main__':
    main()