# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary

# ailever modules
from dataset import AileverDataset
import options


class AileverModule(nn.Module):
    def __init__(self, options):
        super(AileverModule, self).__init__()
        self.identity = nn.Identity()
        self.linear = nn.Linear(9, 9)

    def forward(self, x):
        x = self.identity(x)
        x = self.linear(x)
        return x


def main(options):
    dataset = AileverDataset(options)
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False)
    model = AileverModule(options).to(options.device)
    summary(model, (9, ))

    x_train, y_train = next(iter(dataloader))
    hypothesis = model(x_train)


if __name__ == "__main__":
    options = options.load()
    main(options)
