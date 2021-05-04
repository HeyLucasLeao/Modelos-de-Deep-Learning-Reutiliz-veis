from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)


TOKENIZER = AutoTokenizer.from_pretrained(config['model']['model_name'], do_lower_case=True)

class ShapingDataset(Dataset):

    def __init__(self, texts, targets, max_len):
        super().__init__()
        self.texts = texts
        self.tokenizer = TOKENIZER
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        texts = str(self.texts[item])
        encoding = self.tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        max_length=config['model']['max_seq_length']
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'targets': torch.FloatTensor([self.targets[item]]) 
        }

def create_dataloader(df, max_len, bs, num_workers=4):
    dataset = ShapingDataset(
        texts=df['texts'].to_numpy(),
        targets=df['targets'].to_numpy(),
        max_len=max_len
    )
    data_loader = DataLoader(dataset, bs, num_workers)

    return data_loader