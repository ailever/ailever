## [pytorch](https://pytorch.org/docs/stable/index.html)

## Example, Neural Network
`Installation`
```python
$ pip install torch
```
`Usage`
```python

```

### Neural Network Structure
`Installation`
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

### Neural Network Computational graph
`Installation`
```bash
$ apt install graphviz
$ pip install torchviz
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
