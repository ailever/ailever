# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary

# ailever modules
from dataset import AileverDataset
from modules import AileverModule
import options


class AileverModel(nn.Module):
    def __init__(self, options):
        super(AileverModel, self).__init__()
        self.module = AileverModule(options)
        self.batchnorm = nn.BatchNorm1d(9)
        self.selu = nn.SELU(inplace=True)
        self.linear1 = nn.Linear(9,81)
        self.linear2 = nn.Linear(81,1)

        # wegiht initialize
        nn.init.xavier_normal_(self.linear1.weight, gain=1.0)
        nn.init.xavier_normal_(self.linear2.weight, gain=1.0)
        nn.init.normal_(self.linear1.bias, mean=1.0, std=1.0)
        nn.init.normal_(self.linear1.bias, mean=1.0, std=1.0)


    def forward(self, x):
        x = self.module(x)
        x = self.linear1(self.selu(self.batchnorm(x)))
        x = self.linear2(x)
        return x


def main(options):
    dataset = AileverDataset(options)
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False)
    model = AileverModel(options).to(options.device)
    summary(model, (9, ))

    x_train, y_train = next(iter(dataloader))
    hypothesis = model(x_train)

    

if __name__ == "__main__":
    options = options.load()
    main(options)
