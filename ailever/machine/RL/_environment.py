from abc import ABCMeta, abstractmethod
from torch.distributions.multinomial import Multinomial

class BaseEnvironment(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __update_PR(self):
        pass

    @abstractmethod
    def __update_gymP(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def observe(self):
        pass

    def sampler(self, probs=[0.1, 0.9], size=1):
        total_count = 1
        size = torch.Size([size])
        probs = torch.tensor(probs)
        samples = Multinomial(total_count=total_count, probs=probs).sample(sample_shape=size).squeeze()
	return samples
