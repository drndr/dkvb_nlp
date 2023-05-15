import torch
import datasets
import pandas as pd

from transformers import BertTokenizer

from torch.utils.data import Dataset,DataLoader

# load glue benchmarks into custom datasets
def load_dataset(name, max_len):
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

    def __init__(self, dataframe, tokenizer, max_len, feature1, feature2):
        self.tokenizer = tokenizer
        self.feature1 = dataframe[feature1]
        self.feature2 = dataframe[feature2]
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
      return len(self.feature1)

    def __getitem__(self, index):
        f1 = str(self.feature1[index])
        f1 = " ".join(f1.split())
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
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }