import pandas as pd
import re
from sklearn.model_selection import train_test_split
import csv


class CompanyDatasetFT:

    def __init__(self, path_to_dataset: str, train_size: float = 0.95, train: bool = True,
                 path_to_save: str = '../../data/'):
        self.df = pd.read_csv(path_to_dataset)
        self.train_size = train_size
        self.train = train
        self.path_to_save = path_to_save + 'company.train'
        self.df['name_1'] = self.df['name_1'].apply(lambda x: re.sub(r'[^\w\s]+|[\d]+', r'', x.lower()))
        self.df['name_1'] = self.df['name_1'].apply(lambda x: re.sub(r'[.,"\'-?:!;&]', r'', x.lower()))
        self.df['name_2'] = self.df['name_2'].apply(lambda x: re.sub(r'[^\w\s]+|[\d]+', r'', x.lower()))
        self.df['name_2'] = self.df['name_2'].apply(lambda x: re.sub(r'[.,"\'-?:!;&]', r'', x.lower()))
        self.df['common'] = self.df['name_1'] + " /SEP/  " + self.df['name_2']
        self.df['is_duplicate_str'] = self.df['is_duplicate'].replace({0: '__label__NEGATIVE', 1: '__label__POSITIVE'})
        if self.train == True:
            self.path_to_save = path_to_save + 'company.train'
            self.df, _ = train_test_split(self.df.common, self.df.is_duplicate_str,
                                          train_size=self.train_size,
                                          stratify=self.df.is_duplicate_str)
            self.df[['common', 'is_duplicate_str']].to_csv(self.path_to_save, index=False, sep=' ', header=False,
                                                           escapechar=" ")
        else:
            self.path_to_save = path_to_save + 'company.train'
            _, self.df = train_test_split(self.df.common, self.df.is_duplicate_str,
                                          train_size=self.train_size,
                                          stratify=self.df.is_duplicate_str)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        companies = self.df['is_duplicate_str'].iloc[index] + " " + self.df['name_1'].iloc[index] + " /SEP/ " + \
                    self.df['name_2'].iloc[index]
        return companies
