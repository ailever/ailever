from abc import *

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class BaseNomenclature(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def connect(self):
        pass

class BaseTriggerBlock(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    
    @abstractstatic
    def _dynamic_import():
        pass
    
    @abstractmethod
    def _instance_basis(self):
        pass

    @abstractmethod
    def ui_buffer(self):
        pass

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def loaded_from(self):
        pass

    @abstractmethod
    def store_in(self):
        pass

    @abstractmethod
    def prediction(self):
        pass
 
    @abstractmethod
    def ModelTransferCore(self):
        pass


class BaseTriggerBridge(metaclass=ABCMeta):
    @abstractmethod
    def initializing_local_model_registry(self):
        pass

    @abstractmethod
    def initializing_remote_model_registry(self):
        pass

    @abstractmethod
    def load_from_local_feature_store(self):
        pass

    @abstractmethod
    def load_from_local_source_repository(self):
        pass

    @abstractmethod
    def load_from_local_model_registry(self):
        pass

    @abstractmethod
    def load_from_local_metadata_store(self):
        pass

    @abstractmethod
    def load_from_remote_feature_store(self):
        pass

    @abstractmethod
    def load_from_remote_source_repository(self):
        pass

    @abstractmethod
    def load_from_remote_model_registry(self):
        pass

    @abstractmethod
    def load_from_remote_metadata_store(self):
        pass

    @abstractmethod
    def save_in_local_feature_store(self):
        pass

    @abstractmethod
    def save_in_local_source_repository(self):
        pass

    @abstractmethod
    def save_in_local_model_registry(self):
        pass

    @abstractmethod
    def save_in_local_metadata_store(self):
        pass

    @abstractmethod
    def save_in_remote_feature_store(self):
        pass

    @abstractmethod
    def save_in_remote_source_repository(self):
        pass

    @abstractmethod
    def save_in_remote_model_registry(self):
        pass

    @abstractmethod
    def save_in_remote_metadata_store(self):
        pass
