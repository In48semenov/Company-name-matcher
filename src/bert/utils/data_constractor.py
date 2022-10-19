import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import transformers


class CompanyDataset(Dataset):

    def __init__(self, path_to_dataset: str, tokenizer: transformers, train_size: float = 0.95, train: bool = True):
        df = pd.read_csv(path_to_dataset)

        if train:
            self.df, _ = train_test_split(df, train_size=train_size, stratify=df['is_duplicate'], random_state=17)
        else:
            _, self.df = train_test_split(df, train_size=train_size, stratify=df['is_duplicate'], random_state=17)

        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        companies = '[CLS] ' + self.df['name_1'].iloc[index] + ' [SEP] ' + self.df['name_2'].iloc[index] + ' [SEP]'
        tokens = self.tokenizer(
            companies,
            truncation=True,
            add_special_tokens=False,
            max_length=60,
            pad_to_max_length=True,
            return_tensors='pt'
        )

        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'label': torch.tensor(self.df['is_duplicate'].iloc[index])
        }
