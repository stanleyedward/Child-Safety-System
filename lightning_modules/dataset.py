import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from lightning_modules import config

class ogDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len: int, eval_mode: bool = False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.comment_text
        self.eval_mode = eval_mode 
        if self.eval_mode is False:
            self.targets = self.data.is_toxic
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        output = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
                
        if self.eval_mode is False:
            output['targets'] = torch.tensor(self.targets[index], dtype=torch.float)
                
        return output
    
class lmaoDataset(L.LightningDataModule):
    def __init__(self, datadir, batch_size, num_workers, tokenizer, max_len:int, eval_mode: bool = False):
        self.dataframe = pd.read_csv(datadir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.eval_mode = eval_mode
        self.prepare_data_per_node = True
        self._log_hyperparams = True
        self.allow_zero_length_dataloader_with_multiple_devices = True
        
    def prepare_data(self):
        ogDataset(dataframe=self.dataframe, tokenizer=self.tokenizer, max_len=self.max_len, eval_mode=self.eval_mode)
        # pass
        
    def setup(self, stage = None):
        entire_dataset = ogDataset(dataframe=self.dataframe, tokenizer=self.tokenizer, max_len=self.max_len, eval_mode=self.eval_mode)
        self.train_set, self.val_set = random_split(entire_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.val_set, batch_size = self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    
# if __name__ == '__main__':
#     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
#     dm = LightningDataset(datadir='jigsaw-toxic-comment-classification-challenge/clean_train_data.csv',
#                           batch_size=config.BATCH_SIZE,
#                           tokenizer=tokenizer,
#                           max_len=config.MAX_LEN,
#                           num_workers=config.NUM_WORKERS
#                           )
#     dm.prepare_data()
#     dm.setup()
#     dm.train_dataloader[0]