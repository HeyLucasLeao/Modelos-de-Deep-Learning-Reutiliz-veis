import numpy as np
import torch
import yaml
from tqdm import tqdm

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

def accuracy(outputs, targets):
    softmax = torch.nn.Softmax(dim=-1)
    predictions = softmax(outputs)
    predictions = torch.argmax(outputs, axis=1)
    res = []
    for i in range(len(predictions)):
        if predictions[i] == targets[i]:
            res.append(1)
        else:
            res.append(0)
    res = sum(res) / len(res)

    return res


def train_epoch(
                model, 
                data_loader, 
                criterion, 
                optimizer, 
                device, 
                scheduler):

    model.train()
    losses = []
    acc = []
    for data in tqdm(data_loader, leave=False):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = torch.flatten(data['targets'].to(device))

        outputs = model(
            input_ids=input_ids.squeeze(),
            attention_mask=attention_mask.squeeze()
            )

        #função de perda
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        #Train Accuracy
        acc.append(accuracy(outputs, targets))

        #Back Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #atualiza o learning rate
        scheduler.step()

    avg_loss = np.mean(losses)
    acc = sum(acc) / len(acc)

    return avg_loss, acc

def eval_model(
            model, 
            data_loader, 
            criterion, 
            device):

    model.eval()
    losses = []
    acc = []
    with torch.no_grad():
        for data in data_loader:

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = torch.flatten(data['targets'].to(device))

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
                )

            #função de perda
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            #Eval Accuracy
            acc.append(accuracy(outputs, targets))
            
    acc = sum(acc) / len(acc)
    avg_loss = np.mean(losses)
    return avg_loss, acc