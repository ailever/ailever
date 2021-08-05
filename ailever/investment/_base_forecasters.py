from .__base_structures import BaseForecaster
from ._base_transfer import ModelTransferCore
from .fmlops_nomenclatures import F_MRN

import torch

local_environment = dict()
local_environment['model_registry'] = '.model_registry'
local_environment['model_loading_path'] = '.model_registry'
local_environment['model_saving_path'] = '.model_registry'


class TorchForecaster(BaseForecaster):
    def __init__(self, training_info:dict, local_environment:dict=local_environment, remote_environment:dict=None):
        self.local_environment = local_environmnet
        self.remote_environment = remote_environment

        self.training_info = training_info
        self.model_loading_path = 
        self.model_saving_path = 

        self.load_from_model_registry()
        self.train()
        self.prediction() 

    def initializing_local_model_registry(self):
        saving_directory = self.local_environment['model_registry']
        saving_file = self.training_info['model_name']
        os.path.isdir(saving_directory)
        
        if not os.path.isdir(saving_directory):
            os.mkdir(saving_directory)

        if os.path.isdir(saving_directory):
            if os.path.isfile(os.path.join(saving_directory, saving_file)):
                checkpoint = torch.load('.models/' + training_info['saving_file'])
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                training_info['first'] = checkpoint['first']
                training_info['cumulative_epochs'] = checkpoint['cumulative_epochs']

    def initializing_remote_model_registry(self):
        pass

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_local_model_specification(self):
        pass

    def load_from_local_model_registry(self):
        pass

    def save_in_local_model_registry(self):
        pass

    def load_from_remote_model_specification(self):
        pass

    def load_from_remote_model_registry(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def for_ModelTransferCore(self):
        pass



class TensorflowForecaster(BaseForecaster):
    def __init__(self, training_info:dict, local_environment:dict=local_environment, remote_environment:dict=None):
        pass

    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_local_model_specification(self):
        pass

    def load_from_local_model_registry(self):
        pass

    def save_in_local_model_registry(self):
        pass

    def load_from_remote_model_specification(self):
        pass

    def load_from_remote_model_registry(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def for_ModelTransferCore(self):
        pass



class SklearnForecaster(BaseForecaster):
    def __init__(self, training_info:dict, local_environment:dict=local_environment, remote_environment:dict=None):
        pass

    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_local_model_specification(self):
        pass

    def load_from_local_model_registry(self):
        pass

    def save_in_local_model_registry(self):
        pass

    def load_from_remote_model_specification(self):
        pass

    def load_from_remote_model_registry(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def for_ModelTransferCore(self):
        pass


class StatsmodelsForecaster(BaseForecaster):
    def __init__(self, training_info:dict, local_environment:dict=local_environment, remote_environment:dict=None):
        pass

    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_local_model_specification(self):
        pass

    def load_from_local_model_registry(self):
        pass

    def save_in_local_model_registry(self):
        pass

    def load_from_remote_model_specification(self):
        pass

    def load_from_remote_model_registry(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def for_ModelTransferCore(self):
        pass



