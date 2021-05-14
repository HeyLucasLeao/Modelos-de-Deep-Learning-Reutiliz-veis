from torch import nn, flatten
from transformers import AutoModel
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

MODEL = AutoModel.from_pretrained(config['model']['model_name'])
criterion = nn.CrossEntropyLoss()


for param in MODEL.parameters():
    MODEL.eval()
    param.requires_grad = False

class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.pretrained_model = MODEL
        self.linear1 = nn.Linear(
            98304,
            out_features=1000
        )
        self.linear2 = nn.Linear(
            in_features=1000,
            out_features=500
        )
        self.linear3 = nn.Linear(
            in_features=500,
            out_features=config['model']['n_classes']
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        output = self.pretrained_model.forward(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        output = flatten(output.last_hidden_state, start_dim=1)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear3(output)
        return output

