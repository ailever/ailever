import torch
from visdom import Visdom

obj = type('obj', (), {})


class Visualizer(Visdom):
    def __init__(self):
        super(Visualizer, self).__init__(server='http://localhost',
                                         port=8097,
                                         env='main')
        self._origin = 0
    
    def originator_(self, feature_dim):
        self._origin += 1

        setattr(self, f'window{self._origin}', obj())
        setattr(self, f'window{self._origin}', self.line(Y=torch.Tensor(1, feature_dim).zero_(), opts=dict(title=f'METRIC{feature_dim}' if feature_dim == 1 else 'METRIC',
                                                                                                   xlabel='Epoch',
                                                                                                   ylabel=f'Scale{feature_dim}' if feature_dim == 1 else 'Scale')))
        return getattr(self, f'window{self._origin}')

    

def visualizer(dataset):
    dataset_size = int(dataset.size(0))
    feature_dim = int(dataset.size(1))
    
    vis = Visualizer()
    vis.close(env='main')
    
    # window generator
    window = vis.originator_(feature_dim)
    if feature_dim > 1:
        for i in range(feature_dim):
            locals()[f'sub_window{i}'] = vis.originator_(1)
        
    for i, data in enumerate(dataset):
        vis.line(X=torch.Tensor([[i]*feature_dim]), Y=data.unsqueeze(0), win=window, update='append', opts=dict(title='metrics'))
        
        if feature_dim > 1:
            for f_dim in range(feature_dim):
                vis.line(X=torch.Tensor([[i]]), Y=data[f_dim].unsqueeze(0).unsqueeze(0), win=locals()[f'sub_window{f_dim}'], update='append', opts=dict(title=f'metric{f_dim}'))


x = torch.Tensor(1000,1)
visualizer(x)
