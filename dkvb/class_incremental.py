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
from datasets import load_dataset

from dkv_bn import DiscreteKeyValueBottleneck
from utils import load_glue_dataset, load_cls_dataset, load_class_increment_20ng, load_class_increment_r8, load_class_increment_r52, load_wikipedia
from model import BERTwithBottleNeck, BERTbase, BERT_with_EWC, BERT_with_DERpp

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
            #targets = data['targets'].to(device, dtype=torch.long)
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
  
def train_model_with_DER(start_epoch, n_epochs, training_loader, validation1_loader, model, optimizer, criterion, device, wandb_enabled):
  
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
        outputs = model(ids, mask, token_type_ids, targets, add_buffer=True)
        #Backward
        loss = criterion(outputs, targets)
        
        if not model.buffer.is_empty():
            #print("buffer magic")
            buf_inputs, buf_labels,buf_logits, buf_segment_ids,buf_input_mask = model.buffer.get_data(
                    16)

            buff_inputs = buf_inputs.to(device, dtype=torch.long)
            buff_segment = buf_segment_ids.to(device, dtype=torch.long)
            buff_mask = buf_input_mask.to(device, dtype=torch.long)
            buff_labels = buf_labels.to(device, dtype=torch.long)
            buff_logits = buf_logits.to(device)
            
            outputs_buffer = model(buff_inputs, buff_mask, buff_segment, buff_labels, add_buffer=False)
            
            loss += model.beta * criterion(outputs_buffer, buf_labels)
            loss += model.alpha * model.mse(outputs_buffer, buf_logits)
            
        loss.backward(retain_graph=True)
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
                outputs = model(ids, mask, token_type_ids, targets, add_buffer=False)

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
    parser.add_argument("dataset", default="r8", type=str, help="Dataset: r8, r52")
    parser.add_argument("model", default="base", type=str, help="Type of model (base, dkvb, derpp, ewc)")
    parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training and testing")
    parser.add_argument("--lr_global", default=1e-2, type=float, help="Learning rate for decoder")
    parser.add_argument("--lr_values", default=1e-2, type=float, help="Learning rate for values in bottleneck")
    parser.add_argument("--decoder", default="softmax", type=str, help="Decoder type (mlp, softmax)")
    parser.add_argument("--pooling", default="mean", type=str, help="Type of poolings (cls, mean)")
    parser.add_argument("--key_init", default="full", type=str, help="Type of key init (full, inc, wiki)")
    parser.add_argument("--pool_before", action="store_true", help="enable pooling before bottleneck")
    parser.add_argument("--wandb_enabled", action="store_true", help="wandb monitoring")

    args = parser.parse_args()

    if args.wandb_enabled:
        wandb.init(project="R8 increment", entity="drndr21", name="bert bottleneck softmax")
        #wandb.init(project="R8 increment", entity="drndr21", name=str(args.epochs)+"e "+str(args.batch_size)+"b "+str(args.lr_global)+"Global "+str(args.lr_values)+"Values "+args.pooling+"Ptype ")
    
    # Load to Dataset
    if args.dataset == "r52":
        class_sets = load_class_increment_r52()
        full_train, val_set, n_classes = load_cls_dataset("R52", max_len=256)
    elif args.dataset == "r8":
        class_sets = load_class_increment_r8()
        full_train, val_set, n_classes = load_cls_dataset("R8", max_len=256)
    
        
    # Create validation set here (is reused for all class increment testing)
    validation_params = {'batch_size': args.batch_size,'shuffle': True}             
    validation_loader = DataLoader(val_set, **validation_params)
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    
    if args.model == "dkvb":
        model = BERTwithBottleNeck(args.decoder, 12, args.pool_before, args.pooling, n_classes)
        optimizer = torch.optim.AdamW([{"params": model.enc_with_bottleneck.parameters(), "lr":args.lr_values},
                                       {"params": model.l3.parameters()}],lr=args.lr_global)
    elif args.model == "ewc":
        model = BERT_with_EWC(n_classes, True)
        optimizer = torch.optim.AdamW(model.parameters(),lr =args.lr_global)
    elif args.model == "derpp":
        model = BERT_with_DERpp(n_classes, True)
        optimizer = torch.optim.AdamW(model.parameters(),lr =args.lr_global)
    else:
        model = BERTbase(n_classes, True)
        optimizer = torch.optim.AdamW(model.parameters(),lr =args.lr_global)
    
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)    
    
    train_params = {'batch_size': args.batch_size, 'shuffle': True}   
    
    if args.model == "dkvb" and args.key_init == "full":
        start = timer() # Start measuring time for Key Init, Train and Inference
        training_loader = DataLoader(full_train, **train_params)
        trained_model = discrete_key_init(3,training_loader, model, device)
        end = timer() # Stop measuring time for Key Init
        print("Key Init oracle time in minutes: ",(end-start)/60)
    elif args.model == "dkvb" and args.key_init == "wiki":        
        start = timer() # Start measuring time for Key Init, Train and Inference
        full_train = load_wikipedia(256)
        training_loader = DataLoader(full_train, **train_params)
        trained_model = discrete_key_init(1,training_loader, model, device)
        end = timer() # Stop measuring time for Key Init
        print("Key Init wiki in minutes: ",(end-start)/60)
    elif args.model != "dkvb" or args.key_init == "inc":
        trained_model = model
    
    
    if args.wandb_enabled:
        wandb.watch(model)
    
    # Class incremental training and testing loop
    start_epoch = 1
    end_epoch = args.epochs
    with open(f"{args.dataset}_{args.model}_{args.key_init}.txt", 'w') as file:
        start = timer() # Start measuring time Train and Inference
        for i in range(len(class_sets)):
        
            train_set = class_sets[i]
            
            sampler = RandomSampler(train_set, replacement=True, num_samples=100)
            train_params = {'sampler':sampler,
                            'batch_size': args.batch_size,
                            'shuffle': False
                           }   
            training_loader = DataLoader(train_set, **train_params)
            
            if args.model == "dkvb" and args.key_init == "inc":
                trained_model = discrete_key_init(3,training_loader, trained_model, device)
                
                
            if args.model == "derpp":    
                trained_model = train_model_with_DER(start_epoch, end_epoch, training_loader, validation_loader, trained_model, optimizer, criterion, device, args.wandb_enabled)
            else:
                trained_model = train_model(start_epoch, end_epoch, training_loader, validation_loader, trained_model, optimizer, criterion, device, args.wandb_enabled)
            
            
            if args.model == "ewc":
                trained_model.ewc(i, training_loader, device)
            
            start_epoch = start_epoch+args.epochs
            end_epoch = end_epoch+args.epochs
        
            outputs_matched, targets_matched = test(validation_loader, trained_model, device)
        
            targets_matched=np.array(targets_matched).astype(int)
            outputs_matched=np.array(outputs_matched).astype(int)
            accuracy_matched = metrics.accuracy_score(targets_matched, outputs_matched)
            file.write(f'{accuracy_matched}\n')
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