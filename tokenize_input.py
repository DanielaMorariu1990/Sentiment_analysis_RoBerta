# Importing the libraries needed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import logging
logging.basicConfig(level=logging.INFO)

logger=logging.getLogger("tokenizer_script")

class TokenizeData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.text=dataframe.Phrase
        self.target=dataframe.Sentiment

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text=str(self.text[idx])
        text=" ".join(text.split())

        inputs=self.tokenizer.encode_plus(
            text,
            add_special_tokens = None,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        input_ids=inputs['inputs_ids']
        mask=inputs["attention_mask"]
        token_type_ids=inputs["token_type_ids"]

        return {
            'ids': torch.tensor(input_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float)
        }

def create_train_validation_set(dataframe, train_size=0.8):
    train_df=dataframe.sample(frac=train_size,random_state=43)
    test_df=dataframe.drop(train_df.index).reset_index(drop=True)
    train_df.reset_index(drop=True)
    return train_df, test_df



if __name__=="__main__":
    data=pd.read_csv("train.tsv", delimiter='\t')
    data=data[['Phrase', 'Sentiment']].copy()
    train_df, test_df=create_train_validation_set(dataframe=data, train_size=0.8)
    logger.info("FULL Dataset: {}".format(data.shape))
    logger.info("TRAIN Dataset: {}".format(train_df.shape))
    logger.info("TEST Dataset: {}".format(test_df.shape))
    tokenizer= RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    training_set = TokenizeData(train_df, tokenizer, max_len=256)
    testing_set = TokenizeData(test_df, tokenizer, max_len=256)
    



