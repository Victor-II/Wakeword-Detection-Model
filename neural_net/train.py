import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import CNNModel
from dataset import WakeWordData
from torcheval.metrics import BinaryAccuracy
import pandas as pd
import time
import os


def binary_accuracy(outputs, labels, device):
    outputs = outputs.squeeze()
    metric = BinaryAccuracy().to(device)
    metric.update(outputs, labels)
    return metric.compute()

def create_dl(data, batch_size):
    data_loader = DataLoader(data, batch_size, shuffle=True) 
    return data_loader

def train_single_epoch(model, data_loader, criterion, optimiser, device):
    train_loss, train_accuracy = 0.0, 0.0

    for i, (batch, labels) in enumerate(data_loader):
        batch, labels = batch.to(device), labels.float().to(device)
        outputs = model(batch)
        loss = criterion(outputs.squeeze(), labels)
        accuracy = binary_accuracy(outputs, labels, device)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        train_loss += loss
        train_accuracy += accuracy

    train_loss = train_loss / float(i)
    train_accuracy = train_accuracy / float(i)

    return train_loss, train_accuracy

def evaluate(model, data_loader, criterion, device):
    val_loss, val_accuracy = 0.0, 0.0
    with torch.no_grad():
        for i, (batch, labels) in enumerate(data_loader, start=1):
            batch, labels = batch.to(device), labels.float().to(device)

            outputs = model(batch)
            loss = criterion(outputs.squeeze(), labels)
            accuracy = binary_accuracy(outputs, labels, device)

            val_loss += loss
            val_accuracy += accuracy
        
        val_loss = val_loss / float(i)
        val_accuracy = val_accuracy / float(i)

    return val_loss, val_accuracy


def train(model, criterion, optimiser, epochs, train_dl, val_dl, save_path, device):
    best_loss = 100
    train_losses, train_accuracys, val_losses, val_accuracys = [], [], [], []
    start = time.time()
    print('Training started...\n')
    for epoch in range(epochs):
        start_epoch = time.time()
        train_loss, train_accuracy = train_single_epoch(model, train_dl, criterion, optimiser, device)
        val_loss, val_accuracy = evaluate(model, val_dl, criterion, device)
        if val_loss < best_loss:
            print('Saving model state dict...', end=' ')
            try:
                os.remove('C:/Users/Victor/Desktop/crdmProiect/wakeword/saved_models/cnn_model.pt')
            except:
                pass
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'State dict saved at "{save_path}"')

        train_losses.append(train_loss.cpu().detach().numpy())
        train_accuracys.append(train_accuracy.cpu().detach().numpy())
        val_losses.append(val_loss.cpu().detach().numpy())
        val_accuracys.append(val_accuracy.cpu().detach().numpy())

        elapsed_epoch = time.time() - start_epoch
        print(f'Epoch : [{epoch+1}/{epochs}], {elapsed_epoch / 60.0:.2f} [min]  |  Loss: {train_loss:.5f}  |  Accuracy: {train_accuracy:.5f}  |  Val_loss: {val_loss:.5f}  |  Val_accuracy: {val_accuracy:.5f}')
    elapsed = time.time() - start
    print('---------------------------------------------------------------------')
    print(f'Finished training in {elapsed / 60.0:.2f} [min]')

    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracys,
        'val_loss': val_losses,
        'val_accuracy': val_accuracys
    }

    return history


def main(args):
    train_ds = WakeWordData(args.get('train_json'), args.get('sample_rate'), args.get('num_samples'))
    test_ds = WakeWordData(args.get('test_json'), args.get('sample_rate'), args.get('num_samples'), validation=True)

    train_dl = create_dl(train_ds, args.get('batch_size'))
    val_dl = create_dl(test_ds, args.get('batch_size'))

    model = CNNModel().to(args.get('device'))
    criterion = nn.BCELoss()
    optimiser = Adam(model.parameters(), lr=args.get('learning_rate'))
    
    history = train(model, criterion, optimiser, args.get('epochs'), train_dl, val_dl, args.get('save_state_dict'), args.get('device'))

    return history

if __name__ == '__main__':

    args = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'train_json': 'Path/To/json/train.json',
        'test_json': 'Path/To/json/test.json',
        'save_state_dict': 'Path/To/saved_models/cnn_model.pt',
        'sample_rate': 8000,
        'num_samples': 24000,
        'batch_size': 64,
        'learning_rate': 0.001,
        'epochs': 5
    }

    history = main(args)
    history_df = pd.DataFrame.from_dict(history)
    history_df.to_csv('Path/To/history.csv')
