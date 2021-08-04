from ._base_transfer import ModelTransferCore

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
    def load_from_model_specification(self):
        pass

    @abstractmethod
    def load_from_model_registry(self):
        pass

    @abstractmethod
    def save_in_model_registry(self):
        pass

    @abstractmethod
    def for_ModelTransferCore(self):
        pass



class TorchForecaster(BaseForecaster):
    def __init__(self, training_info, model_registry, model_loading_path=None, model_saving_path=None):
        self.training_info = training_info
        self.model_registry = model_registry
        self.model_registry = model_saving_path

        self.load_from_model_registry()
        self.train()
        self.prediction() 

    def initialize(self):
        pass

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_model_specification(self):
        # from script.py
        pass

    def load_from_model_registry(self):
        self.model = None
        # from .pt
        pass

    def save_in_model_registry(self):
        pass

    def for_ModelTransferCore(self):
        pass



class TensorflowForecaster(BaseForecaster):
    def __init__(self, training_info, model_loading_path, model_saving_path):
        pass

    def initialize(self):
        pass

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_model_specification(self):
        # from script.py
        pass

    def load_from_model_registry(self):
        pass

    def save_in_model_registry(self):
        pass

    def for_ModelTransferCore(self):
        pass



class SklearnForecaster(BaseForecaster):
    def __init__(self, training_info, model_loading_path, model_saving_path):
        pass

    def initialize(self):
        pass

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_model_specification(self):
        # from script.py
        pass

    def load_from_model_registry(self):
        pass

    def save_in_model_registry(self):
        pass

    def for_ModelTransferCore(self):
        pass



class StatsmodelsForecaster(BaseForecaster):
    def __init__(self, training_info, model_loading_path, model_saving_path):
        pass

    def initialize(self):
        pass

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_model_specification(self):
        # from script.py
        pass

    def load_from_model_registry(self):
        pass

    def save_in_model_registry(self):
        pass

    def for_ModelTransferCore(self):
        pass





