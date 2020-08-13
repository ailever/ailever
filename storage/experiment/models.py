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
        self.linear1 = nn.Linear(28*28, 28)
        self.linear2 = nn.Linear(28, 10)
        self.batchnorm = nn.BatchNorm1d(28)
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=-1)

        # wegiht initialize
        nn.init.xavier_normal_(self.linear1.weight, gain=1.0)
        nn.init.xavier_normal_(self.linear2.weight, gain=1.0)
        nn.init.normal_(self.linear1.bias, mean=1.0, std=1.0)
        nn.init.normal_(self.linear1.bias, mean=1.0, std=1.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(self.selu(self.batchnorm(x)))
        x = self.softmax(x)
        return x


def main(options):
    train_dataset, _, _ = AileverDataset(options)
    train_dataloader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=False)
    model = AileverModel(options).to(options.device)
    summary(model, (28*28, ))

    x_train, y_train = next(iter(train_dataloader))
    hypothesis = model(x_train.view(options.batch_size, 28*28).to(options.device))

    

if __name__ == "__main__":
    options = options.load()
    main(options)
