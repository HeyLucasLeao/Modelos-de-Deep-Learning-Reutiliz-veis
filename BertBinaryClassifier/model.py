import torch
from transformers import AutoModel
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

MODEL = AutoModel.from_pretrained(config['model']['model_name'])
criterion = torch.nn.BCELoss()


for param in MODEL.parameters():
    MODEL.eval()
    param.requires_grad = False

class Classifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained_model = MODEL
        self.linear1 = torch.nn.Linear(
            98304,
            out_features=1000
        )
        self.linear2 = torch.nn.Linear(
            in_features=1000,
            out_features=500
        )
        self.linear3 = torch.nn.Linear(
            in_features=500,
            out_features=1
        )
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output = self.pretrained_model.forward(
            input_ids = input_ids.squeeze(),
            attention_mask = attention_mask.squeeze()
        )
        output = torch.flatten(output.last_hidden_state, start_dim=1)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear3(output)
        output = self.sigmoid(output)
        return output

