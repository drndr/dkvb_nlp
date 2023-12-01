import torch
import datasets
import pandas as pd
import os.path as osp
from tqdm import tqdm
from collections import Counter


from transformers import BertTokenizer, RobertaTokenizer

from torch.utils.data import Dataset,DataLoader


def load_cls_dataset(name, max_len):
    
    print("Loading raw documents")
    with open(osp.join('datasets_cls',name, name+'_raw.txt'), 'rb') as f:
        raw_documents = [line.strip().decode('latin1') for line in tqdm(f)]
    
    train_labels = []
    test_labels = []
    train_data = []
    test_data = []

    print("Loading document metadata...")
    doc_meta_path = osp.join('datasets_cls',name, name+'_meta.txt')
    with open(doc_meta_path, 'r') as f:
        i=0
        for idx, line in tqdm(enumerate(f)):
            __name, train_or_test, label = line.strip().split('\t')
            if 'test' in train_or_test:
                test_labels.append(label)
                test_data.append(raw_documents[i])
            elif 'train' in train_or_test:
                train_labels.append(label)
                train_data.append(raw_documents[i])
            else:
                raise ValueError("Doc is neither train nor test:"+ doc_meta_path + ' in line: ' + str(idx+1))
            i+=1
            
    print("Encoding labels...")
    label2index = {label: idx for idx, label in enumerate(sorted(set([*train_labels, *test_labels])))}
    train_label_ids = [label2index[train_label] for train_label in tqdm(train_labels)]
    test_label_ids = [label2index[test_label] for test_label in tqdm(test_labels)]

    train_labels = train_label_ids
    train_data = train_data
    test_labels = test_label_ids
    test_data = test_data    
    
    train_df = pd.DataFrame()
    train_df['text'] = train_data
    train_df['label'] = train_labels

    test_df = pd.DataFrame()
    test_df['text'] = test_data
    test_df['label'] = test_labels

    print("Number of train texts ",len(train_df['text']))
    print("Number of train labels ",len(train_df['label']))
    print("Number of test texts ",len(test_df['text']))
    print("Number of test labels ",len(test_df['label']))
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    
    train_set =  CustomDataset(train_df, tokenizer, max_len, 'text', None)
    val_set = CustomDataset(test_df, tokenizer, max_len, 'text', None)
    
    n_classes = len(set([*train_labels, *test_labels]))
    print("Number of classes ", n_classes)
    
    return train_set, val_set, n_classes
    


# Create class incremental subsets from dataset (customized for R8)        
def load_class_increment():
    full_set = load_cls_dataset("R8",256)[0]
    subset_list = []
    for value in range(8):
        subset_indices = []
        for index in range(len(full_set)):
            # Access the field value for the current sample
            sample = full_set[index]
            class_value = sample["targets"]

            # Check if the field value matches the desired value
            if class_value == value:
                subset_indices.append(index)

        # Create a Subset of the original dataset using the subset_indices
        subset = torch.utils.data.Subset(full_set, subset_indices)
        print("Class ",value," size ", len(subset))
        subset_list.append(subset)
    return subset_list
    
    
    
# load glue benchmarks into custom datasets
def load_glue_dataset(name, max_len):
    dataset = datasets.load_dataset(path='glue', name=name)
    
    print("Split names in ",name)
    for k in dataset:
        print('   ', k)
    
    features = list(dataset['train'].features.keys())
    print('\nThe feature names in ', name,' dataset:', features, '\n')
    
    n_classes = len(dataset['train'].features['label'].names)
    print('The class names in ', name,' dataset:', dataset['train'].features['label'].names, '\n')
    
    print('The number of training examples in ', name,' dataset:', dataset['train'].num_rows)
    
    # MNLI has two validation and test sets: matched, mismatched
    if name != 'mnli':
        print('The number of validation examples in ', name,' dataset', dataset['validation'].num_rows)
        print('The number of testing examples in ', name,' dataset', dataset['test'].num_rows, '\n')
        
        validation_df = pd.DataFrame(dataset['validation'])
    else:
        print("MNLI validation set: Matched")
        print('The number of validation matched examples in ', name,' dataset', dataset['validation_matched'].num_rows)
        print('The number of testing examples in ', name,' dataset', dataset['test_matched'].num_rows, '\n')
        
        validation_df = pd.DataFrame(dataset['validation_matched'])
    
    train_df = pd.DataFrame(dataset['train'])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    train_set =  CustomDataset(train_df, tokenizer, max_len, features[0], features[1])
    val_set = CustomDataset(validation_df, tokenizer, max_len, features[0], features[1])
    
    return train_set, val_set, n_classes            

# Custom Dataset for text pair inputs: <cls> text1 <sep> text2    
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, feature1, feature2=None):
        self.tokenizer = tokenizer
        self.feature1 = dataframe[feature1]
        if feature2 is not None:
            self.feature2 = dataframe[feature2]
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
      return len(self.feature1)

    def __getitem__(self, index):
        f1 = str(self.feature1[index])
        f1 = " ".join(f1.split())
        if hasattr(self, 'feature2'):
        
            f2 = str(self.feature2[index])
            f2 = " ".join(f2.split())
        
            inputs = self.tokenizer.encode_plus(
                f1,
                f2,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_token_type_ids=True,
                verbose = False
            )
        else:
        
            inputs = self.tokenizer.encode_plus(
                f1,
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