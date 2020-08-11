# built-in / external modules
import os

# torch
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary

# ailever modules
from datasets import AileverDataset
from models import AileverModel
import options


def train(options):
    # dataset
    dataset = AileverDataset(options)
    train_dataloader = DataLoader(dataset.type('train'), batch_size=options.batch_size, shuffle=True)
    validation_dataloader = DataLoader(dataset.type('valiadtion'), batch_size=options.batch_size, shuffle=False)
    
    # model
    model = AileverModel(options).to(options.device)
    criterion = nn.MSELoss().to(options.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
    summary(model, (9, ))

    epochs = options.epochs
    for epoch in range(epochs):
        # Training
        for batch_idx, (x_train, y_train) in enumerate(train_dataloader):
            # forward
            hypothesis = model(x_train)
            cost = criterion(hypothesis, y_train)
            
            # backward
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            # visualization
            html = f'<b>[TRAINING][{batch_idx+1}/{len(train_dataloader)}]</b> <br>* SIZE : {x_train.size()} <br>* INPUT : {x_train[0].data} <br>* TURE : {y_train[0].data} <br>* PRED : {hypothesis[0].data}'
            options.vis.visualize(epoch, x=batch_idx, y=cost.data, mode='train', html=html)
        print(f'[TRAINING][Epoch:{epoch+1}/{epochs}] : Loss = {cost}')
        print(f'- Prediction : {x_train[0].data} >> {hypothesis[0].data}')
        

        # Validation
        with torch.no_grad():
            model.eval()
            for batch_idx, (x_train, y_train) in enumerate(validation_dataloader):
                # forward
                hypothesis = model(x_train)
                cost = criterion(hypothesis, y_train)
                
                # visualization
                html = f'<b>[VALIDATION][{batch_idx+1}/{len(validation_dataloader)}]</b><br>* SIZE : {x_train.size()} <br>* INPUT : {x_train[0].data} <br>* TURE : {y_train[0].data} <br>* PRED : {hypothesis[0].data}'
                options.vis.visualize(epoch, x=batch_idx, y=cost.data, mode='validation', html=html)
            print(f'[VALIDATION][Epoch:{epoch+1}/{epochs}] : Loss = {cost}')
            print(f'- Prediction : {x_train[0].data} >> {hypothesis[0].data}')


    if not os.path.isdir('.Log') : os.mkdir('.Log')
    torch.save(model.state_dict(), f'.Log/model_{options.id}.pth')
    print(f'[AILEVER] The file ".Log/model_{options.id}.pth" is successfully saved!')
 

if __name__ == "__main__":
    options = options.load()
    train(options)
