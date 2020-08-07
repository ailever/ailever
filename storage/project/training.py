# built-in / external modules
import os

# torch
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# ailever modules
from datasets import AileverDataset
from models import AileverModel
import options



def train(options):

    dataset = AileverDataset(options)
    train_dataloader = DataLoader(dataset.type('train'), batch_size=options.batch_size, shuffle=True)
    validation_dataloader = DataLoader(dataset.type('valiadtion'), batch_size=options.batch_size, shuffle=False)
    
    model = AileverModel(options).to(options.device)
    criterion = nn.MSELoss().to(options.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
    
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
            
            print(f'[TRAINING][Epoch:{epoch+1}/{epochs}][Batch:{batch_idx+1}/{len(train_dataloader)}] : Loss = {cost}', end='\r')
        print(f'[TRAINING][Epoch:{epoch+1}/{epochs}] : Loss = {cost}', end='\r')
        

        # Validation
        with torch.no_grad():
            model.eval()
            for batch_idx, (x_train, y_train) in enumerate(validation_dataloader):
                # forward
                hypothesis = model(x_train)
                cost = criterion(hypothesis, y_train)
                
                print(f'[VALIDATION][Epoch:{epoch+1}/{epochs}][Batch:{batch_idx+1}/{len(validation_dataloader)}] : Loss = {cost}', end='\r')
            print(f'[VALIDATION][Epoch:{epoch+1}/{epochs}] : Loss = {cost}', end='\r')


    if not os.path.isdir('.Log') : os.mkdir('.Log')
    torch.save(model.state_dict(), f'.Log/model_{options.id}.pth')
    print(f'[AILEVER] The file ".Log/model_{options.id}.pth" is successfully saved!')
 


if __name__ == "__main__":
    options = options.load()
    train(options)
