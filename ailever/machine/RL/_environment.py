from abc import ABCMeta, abstractmethod
from torch.distributions.multinomial import Multinomial

class BaseEnvironment(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _update_PR(self):
        pass

    @abstractmethod
    def _update_gymP(self):
        pass

    @abstractmethod
    def _ProcessCore(self):
        pass

    @abstractmethod
    def sampler(self):
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
