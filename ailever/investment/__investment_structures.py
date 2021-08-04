from abc import *


class BaseForecaster(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def prediction(self):
        pass
    
    @abstractmethod
    def load_from_local_model_specification(self):
        pass

    @abstractmethod
    def load_from_local_model_registry(self):
        pass

    @abstractmethod
    def save_in_local_model_registry(self):
        pass

    @abstractmethod
    def load_from_remote_model_specification(self):
        pass

    @abstractmethod
    def load_from_remote_model_registry(self):
        pass

    @abstractmethod
    def save_in_remote_model_registry(self):
        pass

    @abstractmethod
    def for_ModelTransferCore(self):
        pass

