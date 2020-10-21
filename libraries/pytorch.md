## [Deep Learning] | [pytorch](https://pytorch.org/docs/stable/index.html) | [github](https://github.com/pytorch/pytorch)


## Example, Neural Network
`Installation`
```python
$ pip install torch
```
`Usage`
```python
import torch
import torch.nn as nn
from torch import optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        x = self.linear(x)
        return x

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, hypothesis, target):
        return self.mse(hypothesis, target)

x_train = torch.arange(0,10).type(torch.FloatTensor).unsqueeze(-1)
target = x_train.mul(5).add(10)

model = Model()
criterion = Criterion()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    hypothesis = model(x_train)
    cost = criterion(hypothesis, target)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100 == 0:
        print(cost)

print(model.linear.weight)
print(model.linear.bias)
```

### Neural Network Structure
`Installation` : [pytorch-summary](https://github.com/sksq96/pytorch-summary)
```bash
$ pip install torch-summary
```
`Usage`
```python
import torch
import torch.nn as nn
from torchsummary import summary

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(2,2)

    def forward(self, x):
        return self.linear(x)

model = Model()
x = torch.Tensor(100,2).uniform_(0,1)
y = model(x)

summary(model, (2,))
```
![image](https://user-images.githubusercontent.com/52376448/95554784-baef4500-0a4b-11eb-96c4-07a10fd3e2c3.png)

### Neural Network Computational Graph
`Installation for linux` : [pytorchviz](https://github.com/szagoruyko/pytorchviz)
```bash
$ apt install graphviz
$ pip install torchviz
```
`Installation for windows` : [graphviz download](https://graphviz.org/download/)
```dos
pip install graphviz
pip install torchviz
```
`Usage`
```python
import torch
import torch.nn as nn
from torchviz import make_dot

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(2,2)

    def forward(self, x):
        return self.linear(x)

model = Model()
x = torch.Tensor(100,2).uniform_(0,1)
y = model(x)

make_dot(y, params=dict(model.named_parameters()))
```
![image](https://user-images.githubusercontent.com/52376448/95554752-ac089280-0a4b-11eb-8955-f23c2e29653e.png)
