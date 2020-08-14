# built-in / external modules
import os
import time

# torch
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary

# ailever modules
from dataset import AileverDataset
from models import AileverModel
import options


def train(options):
    # dataset
    train_dataset, validation_dataset, _ = AileverDataset(options)
    train_dataloader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True, drop_last=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=options.batch_size, shuffle=False, drop_last=False)
    
    # model
    model = AileverModel(options).to(options.device)
    criterion = nn.CrossEntropyLoss().to(options.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
    summary(model, (28*28, ))

    epochs = options.epochs
    for epoch in range(epochs):
        # Training
        for batch_idx, (x_train, y_train) in enumerate(train_dataloader):
            x_train = x_train.view(options.batch_size, 28*28).to(options.device)
            y_train = y_train.to(options.device)

            # forward
            time_start = time.time()
            hypothesis = model(x_train)
            cost = criterion(hypothesis, y_train)
            
            # backward
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            time_end = time.time()
            
            # visualization
            html = f"""<b>{options.dataset_name.upper()}</b><br>
                       <b>[TRAINING][{epoch+1}/{epochs}][{batch_idx+1}/{len(train_dataloader)}]</b> <br>
                       * SIZE : {x_train.size()} <br>
                       * TURE : {y_train[0].data} <br>
                       * PRED : {torch.argmax(hypothesis[0], dim=-1).data} <br>
                       * LOSS : {cost.data} <br>
                       * TIME : {time_end - time_start:.10f}(sec)"""
            options.add.vis.visualize(epoch, x=batch_idx, y=cost.data, mode='train', html=html)
        print(f'[TRAINING][Epoch:{epoch+1}/{epochs}] : Loss = {cost}')
        print(f'- Prediction/True : {torch.argmax(hypothesis[0], dim=-1).data}/{y_train[0].data}')
        

        # Validation
        with torch.no_grad():
            model.eval()
            for batch_idx, (x_train, y_train) in enumerate(validation_dataloader):
                x_train = x_train.view(options.batch_size, 28*28).to(options.device)
                y_train = y_train.to(options.device)
                
                # forward
                time_start = time.time()
                hypothesis = model(x_train)
                cost = criterion(hypothesis, y_train)
                time_end = time.time()
                
                # visualization
                html = f"""<b>{options.dataset_name.upper()}</b><br>
                           <b>[VALIDATION][{epoch+1}/{epochs}][{batch_idx+1}/{len(validation_dataloader)}]</b> <br>
                           * SIZE : {x_train.size()} <br>
                           * TURE : {y_train[0].data} <br>
                           * PRED : {torch.argmax(hypothesis[0], dim=-1).data} <br>
                           * LOSS : {cost.data} <br>
                           * TIME : {time_end - time_start:.10f}(sec)"""
                options.add.vis.visualize(epoch, x=batch_idx, y=cost.data, mode='validation', html=html)
            print(f'[VALIDATION][Epoch:{epoch+1}/{epochs}] : Loss = {cost}')
            print(f'- Prediction/True : {torch.argmax(hypothesis[0], dim=-1).data}/{y_train[0].data}')


    if not os.path.isdir('.Log') : os.mkdir('.Log')
    torch.save(model.state_dict(), f'.Log/model_{options.id}.pth')
    print(f'[AILEVER] The file ".Log/model_{options.id}.pth" is successfully saved!')
 

if __name__ == "__main__":
    options = options.load()
    train(options)
