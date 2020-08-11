import torch
from visdom import Visdom

class AileverVisualizer(Visdom):
    def __init__(self, epochs, titles):
        super(AileverVisualizer, self).__init__(server='http://localhost', port=8097, env='main')
        self.close(env='main')
        self._text = self.text('')
        self._origin = -1
        self._epoch = 0
        
        for epoch in range(epochs):
            # window generator
            self.originator_(epoch, titles)
    
    def visualize(self, epoch, x, y, mode, html):
        # line plot
        if mode == 'train':
            self.text(html, win=self._text)
            self.line(X=torch.Tensor([[x, 0]]), Y=torch.Tensor([[y, 0]]), win=getattr(self, f'window{epoch}'), update='append', opts=dict(title=mode))
        elif mode == 'validation':
            self.text(html, win=self._text)
            self.line(X=torch.Tensor([[0, x]]), Y=torch.Tensor([[0, y]]), win=getattr(self, f'window{epoch}'), update='append', opts=dict(title=mode))

    def originator_(self, epoch, titles):
        setattr(self, f'window{epoch}', self.line(Y=torch.Tensor(1, 2).zero_(), opts=dict(title='LOSS'+f'(epoch {epoch})',
                                                                                                 xlabel=f'BATCH INDEX',
                                                                                                 ylabel=f'SCALE',
                                                                                                 legend=[titles[0], titles[1]],
                                                                                                 showlegend=True)))


def main():
    x = torch.arange(100).type(torch.FloatTensor)
    y = torch.arange(100).type(torch.FloatTensor) + 1
    
    epochs = 5
    vis = AileverVisualizer(epochs, titles=['train', 'validation'])
    for epoch in range(epochs):
        for _x, _y in zip(x,y):
            html = f'{_x} <br> {_y}'
            vis.visualize(epoch, _x, _y, mode='train', html='')


if __name__ == "__main__":
    main()
