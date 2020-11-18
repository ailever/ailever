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

### [forward] : Neural Network Structure 
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

### [backward] : Neural Network Computational Graph
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

### Save and Load
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

model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# [SAVE]
torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 'checkpoint.pth')

# [LOAD]
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```
```bash
$ ls
    > checkpoint.pth
```

<br><br><br>
## Visualization : [visdom](https://github.com/facebookresearch/visdom)
`Installation`
```bash
$ pip install visdom
```
### From local machine,
`http://localhost:[port]`
```bash
$ python -m visdom.server               # default port : 8097
$ python -m visdom.server -p [port]
```
`one figure`
```python
from visdom import Visdom
import torch

vis = Visdom(server='http://localhost', port=8097, env='main')
vis.close(env='main')

# origin
window = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='TITLE'))
graphic_options = dict()
graphic_options['title'] = 'title'
graphic_options['xlabel'] = 'xlabel'
graphic_options['ylabel'] = 'ylabel'
graphic_options['showlegend'] = True

white_noise = torch.Tensor(100).normal_(0, 1)
for t, noise in enumerate(white_noise):
    vis.line(X=torch.tensor([t]), Y=torch.tensor([noise]), win=window, update='append', opts=graphic_options)
```
![image](https://user-images.githubusercontent.com/52376448/96789426-1f3cdc00-1430-11eb-9629-bf57d99594fb.png)

```python
from visdom import Visdom
import torch

vis = Visdom(server='http://localhost', port=8097, env='main')
vis.close(env='main')

# origin
features = 5
window = vis.line(Y=torch.Tensor(1, features).zero_(), opts=dict(title='TITLE'))
graphic_options = dict()
graphic_options['title'] = 'title'
graphic_options['xlabel'] = 'xlabel'
graphic_options['ylabel'] = 'ylabel'
graphic_options['showlegend'] = True

white_noise = torch.Tensor(50, features).normal_(0, 1)
for t, noise in enumerate(white_noise):
    vis.line(X=torch.tensor([[t]*features]), Y=noise.unsqueeze(0), win=window, update='append', opts=graphic_options)
```
![image](https://user-images.githubusercontent.com/52376448/96790463-ce2de780-1431-11eb-9ece-a02d784b75d5.png)
`several figures`
```python
from visdom import Visdom
import torch

vis = Visdom(server='http://localhost', port=8097, env='main')
vis.close(env='main')

# origin
window1 = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='TITLE'))
window2 = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='TITLE'))
graphic_options = dict()
graphic_options['title'] = 'title'
graphic_options['xlabel'] = 'xlabel'
graphic_options['ylabel'] = 'ylabel'
graphic_options['showlegend'] = True

white_noise = torch.Tensor(100).normal_(0, 1)
for t, noise in enumerate(white_noise):
    vis.line(X=torch.tensor([t]), Y=torch.tensor([noise]), win=window1, update='append', opts=graphic_options)
    vis.line(X=torch.tensor([-t]), Y=torch.tensor([noise]), win=window2, update='append', opts=graphic_options)
```
![image](https://user-images.githubusercontent.com/52376448/96791033-b0ad4d80-1432-11eb-9b5b-741c2e89a745.png)
`text, image, images`
```python
from visdom import Visdom
import torch

vis = Visdom(server='http://localhost', port=8097, env='main')
vis.close(env='main')

# Text
vis.text("Hello, world!",env="main")

# Image
a=torch.randn(3,200,200)
vis.image(a)

# Images
vis.images(torch.Tensor(3,3,28,28).normal_(0,1))
```
![image](https://user-images.githubusercontent.com/52376448/96792176-cfacdf00-1434-11eb-9385-ae08a2c6605f.png)

<br><br><br>

### From remote server,
`http://localhost:[port]`<br>
`on remote terminal`
```bash
$ python -m visdom.server -p [port]
$ ssh -N -f -L localhost:[port]:localhost:[port] [id]@$[localhost-ip]
```




## MODELS
### Multi-Layer Perceptron
```python
import torch
import torch.nn as nn
from torch import optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(1,10)
        self.linear2 = nn.Linear(10,10)
        self.linear3 = nn.Linear(10,1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0)
        self.batch_norm = torch.nn.BatchNorm1d(10)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(self.drop(self.relu(x)))
        x = self.linear3(self.drop(self.sigmoid(self.batch_norm(x))))
        return x

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, hypothesis, target):
        return self.mse(hypothesis, target)

x_train = torch.arange(0,100).type(torch.FloatTensor).unsqueeze(-1)
target = x_train.mul(5).add(10)

model = Model()
criterion = Criterion()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 5000
for epoch in range(epochs):
    hypothesis = model(x_train)
    cost = criterion(hypothesis, target)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100 == 0:
        print(cost)
```

### Convolutional Neural Network
### Recurrent Neural Network
#### LSTM
```python
import torch
import torch.nn as nn
from torch import optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(1,16,1, batch_first=True)
        self.linear = nn.Linear(16,1)

    def forward(self, x):
        o, (h, c) = self.lstm(x)
        x = self.linear(h[-1]).squeeze()

        """
        x = (batch, sequence, x_dim)
        o = (batch, sequence, h_dim)
        h = (layer, batch, h_dim)

        x = torch.Size([5, 3, 1])
        o = torch.Size([5, 3, 16])
        h = torch.Size([1, 5, 16])
        """

        return x

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, hypothesis, target):
        return self.mse(hypothesis, target)

x_train = torch.arange(5*3).type(torch.FloatTensor).view(5,3,1)
target = x_train.mean(dim=1).squeeze()
print(x_train)
print(target, '\n')

model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = Criterion()

epochs = 3000
for epoch in range(epochs):
    hypothesis = model(x_train)
    cost = criterion(hypothesis, target)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100 == 0:
        print(cost)
```

#### Attention
```python
import torch
import torch.nn as nn
from torch import optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(3,1)
        self.head_attention = nn.MultiheadAttention(embed_dim=1, num_heads=1)

    def forward(self, x):
        q = x.transpose(0,1)
        Q, w = self.head_attention(q,q,q)
        Q_t = Q.transpose(0,1).squeeze(-1)
        harmonic_x = self.linear(Q_t).squeeze()

        """
        x = (batch, sequence, x_dim)
        q = (sequence, batch, x_dim)
        Q = (sequence, batch, x_dim)
        Q_t = (batch, sequence)

        x = (5,3,1)
        q = (3,5,1)
        Q = (3,5,1)
        Q_t = (5,3)
        """

        return harmonic_x

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, hypothesis, target):
        return self.mse(hypothesis, target)

x_train = torch.arange(5*3).type(torch.FloatTensor).view(5,3,1)
target = x_train.mean(dim=1).squeeze()
print(x_train)
print(target, '\n')

model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = Criterion()

epochs = 3000
for epoch in range(epochs):
    hypothesis = model(x_train)
    cost = criterion(hypothesis, target)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100 == 0:
        print(cost)
```
