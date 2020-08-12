# torch
import torch
from visdom import Visdom

# ailever modules
import options


class AileverVisualizer(Visdom):
    def __init__(self, options):
        super(AileverVisualizer, self).__init__(server=options.server, port=options.port, env=options.env)
        self.titles = ['train', 'validation']
        self.close(env='main')
        self._text = self.text('')
        self._origin = -1
        self._epoch = 0
        
        for epoch in range(options.epochs):
            # window generator
            self.originator_(epoch, self.titles)
    
    def texting(self, html):
        self.text(html, win=self._text)

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


def main(options):
    x = torch.arange(100).type(torch.FloatTensor)
    y = torch.arange(100).type(torch.FloatTensor)
    
    epochs = 5
    vis = AileverVisualizer(options)
    for epoch in range(epochs):
        for _x, _y in zip(x,y):
            html = f'{_x.data} <br> {_y.data}'
            vis.visualize(epoch, _x, _y, mode='train', html=html)
    
    vis.texting('hi,<br>hello')

if __name__ == "__main__":
    options = options.load()
    main(options)
