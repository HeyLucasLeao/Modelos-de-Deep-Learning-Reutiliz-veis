import numpy as np
import torch
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

def train_epoch(
                model, 
                data_loader, 
                criterion, 
                optimizer, 
                device, 
                scheduler):

    model.train()
    losses = []
    train_acc = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['targets'].to(device).view(-1, 1)

        outputs = model(
            input_ids=input_ids.squeeze(),
            attention_mask=attention_mask.squeeze()
            )        
        #função de perda
        predictions = (outputs >= 0.50).type(torch.long)
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        #Eval Accuracy
        train_acc += torch.sum(predictions == targets)
        
        #Back Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #atualiza o learning rate
        scheduler.step()

    total_acc = train_acc/len(data_loader.dataset)
    avg_loss = np.mean(losses)

    return avg_loss,total_acc

def eval_model(
            model, 
            data_loader, 
            criterion, 
            device):

    model.eval()
    losses = []
    eval_acc = 0
    with torch.no_grad():
        for data in data_loader:

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
                )

            #função de perda
            predictions = (outputs >= 0.50).type(torch.long)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            #Eval Accuracy
            eval_acc += torch.sum(predictions == targets)


    total_acc = eval_acc/len(data_loader.dataset)
    avg_loss = np.mean(losses)
    return avg_loss, total_acc