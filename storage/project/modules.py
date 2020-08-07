# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ailever modules
from datasets import AileverDataset
import options


class AileverModule(nn.Module):
    def __init__(self, options):
        super(AileverModule, self).__init__()
        self.linear1 = nn.Linear(5,100)
        self.linear2 = nn.Linear(100,5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def main(options):
    dataset = AileverDataset(options)
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False)
    model = AileverModule(options).to(options.device)

    x_train, y_train = next(iter(dataloader))
    hypothesis = model(x_train)


if __name__ == "__main__":
    options = options.load()
    main(options)
