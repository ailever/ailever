# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary

# ailever modules
from datasets import AileverDataset
from modules import AileverModule
import options


class AileverModel(nn.Module):
    def __init__(self, options):
        super(AileverModel, self).__init__()
        self.module = AileverModule(options)
        self.linear1 = nn.Linear(5,100)
        self.linear2 = nn.Linear(100,5)

    def forward(self, x):
        x = self.module(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def main(options):
    dataset = AileverDataset(options)
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False)
    model = AileverModel(options).to(options.device)
    summary(model, (5, ))

    x_train, y_train = next(iter(dataloader))
    hypothesis = model(x_train)

    

if __name__ == "__main__":
    options = options.load()
    main(options)
