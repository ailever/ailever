from .__base_structures import BaseTriggerBlock
from ._fmlops_policy import fmlops_bs, local_initialization_policy
from ._base_transfer import ModelTransferCore

from importlib import import_module
from functools import partial
import torch

"""
local_environment['feature_store'] = ''
local_environment['source_repository'] = ''
local_environment['model_registry'] = ''
local_environment['metadata_store'] = ''
local_environment['model_loading_path'] = ''
local_environment['model_saving_path'] = ''
"""

class TorchTriggerBlock(BaseTriggerBlock):
    def __init__(self, training_info:dict, local_environment:dict=None, remote_environment:dict=None):
        if local_environment:
            local_initialization_policy(local_environment)
            self.local_environment = local_environment
            self.remote_environment = remote_environment

        self.prediction() 
        self.outcome_report()

    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass
    
    @staticmethod
    def _dynamic_import(architecture, module):
        return getattr(import_module(f'.fmlops_forecasters.torch.{architecture}'), module)

    def _instance_basis(self, train_specification):
        architecutre = train_specification['architecture']
        InvestmentDataLoader = self.dynamic_import(architecture, 'InvestmentDataLoader')
        Model = self.dynamic_import(architecture, 'Model')
        Criterion = self.dynamic_import(architecture, 'Criterion')
        Optimizer = self.dynamic_import(architecture, 'Optimizer')

        train_dataloader, test_dataloader = InvestmentDataLoader(train_specification)
        model = Model(train_specification)
        criterion = Criterion(train_specification)
        optimizer = Optimizer(train_specification)

        # instance update
        checkpoint = torch.load(os.path.join(source_repository['source_repository'], source))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return train_dataloader, test_dataloader, model, criterion, optimizer

    def train(self, train_specification):
        train_dataloader, test_dataloader, model, criterion, optimizer = self._instance_basis(train_specification)
        for epoch in range(epochs):
            training_losses = []
            model.train()
            for batch_idx, (x_train, y_train) in enumerate(train_dataloader):
                # Forward
                hypothesis = model(x_train.squeeze().float().to(training_info['device']))
                cost = criterion(hypothesis.squeeze().float().to(training_info['device']), y_train.squeeze().float().to(training_info['device']))
                # Backward
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                training_losses.append(cost)
            # Alert
            TrainMSE = torch.mean(torch.tensor(training_losses)).data
            if 'cumulative_epochs' in training_info.keys():
                previous_cumulative_epochs = training_info['cumulative_epochs']
                cumulative_epochs = training_info['cumulative_epochs'] + epochs
                print(f'[Training  ][{epoch+1+previous_cumulative_epochs}/{cumulative_epochs}]', float(TrainMSE))
            else:
                print(f'[Training  ][{epoch+1}/{epochs}]', float(TrainMSE))
            
            if torch.isnan(TrainMSE).all():
                break 

            with torch.no_grad():
                validation_losses = []
                model.eval()
                for batch_idx, (x_train, y_train) in enumerate(test_dataloader):
                    # Forward
                    hypothesis = model(x_train.squeeze().float().to(training_info['device']))
                    cost = criterion(hypothesis.squeeze().float().to(training_info['device']), y_train.squeeze().float().to(training_info['device']))
                    validation_losses.append(cost)
                # Alert
                ValidationMSE = torch.mean(torch.tensor(validation_losses)).data
                if 'cumulative_epochs' in training_info.keys():
                    previous_cumulative_epochs = training_info['cumulative_epochs']
                    cumulative_epochs = training_info['cumulative_epochs'] + epochs
                    print(f'[Validation][{epoch+1+previous_cumulative_epochs}/{cumulative_epochs}]', float(ValidationMSE))
                else:
                    print(f'[Validation][{epoch+1}/{epochs}]', float(ValidationMSE))

        self.save_in_local_model_registry()
        self.save_in_remote_model_registry()

    def prediction(self):
        pass

    def load_from_ailever_source_repository(self):
        return None

    def load_from_local_rawdata_repository(self):
        return None

    def load_from_local_feature_store(self):
        return None

    def load_from_local_source_repository(self):
        return None

    def load_from_local_model_registry(self):
        return None

    def load_from_local_metadata_store(self):
        return None

    def remote_from_local_rawdata_repository(self):
        return None

    def remote_from_local_feature_store(self):
        return None

    def remote_from_remote_source_repository(self):
        return None

    def remote_from_remote_model_registry(self):
        return None

    def remote_from_remote_matadata_store(self):
        return None

    def save_in_local_feature_store(self):
        pass

    def save_in_local_source_repository(self):
        pass

    def save_in_local_model_registry(self):
        pass

    def save_in_local_metadata_store(self):
        pass

    def save_in_remote_feature_store(self):
        pass

    def save_in_remote_source_repository(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def save_in_remote_metadata_store(self):
        pass

    def ModelTransferCore(self):
        modelcore = ModelTransferCore()
        return modelcore



class TensorflowTriggerBlock(BaseTriggerBlock):
    def __init__(self, training_info:dict, local_environment:dict=None, remote_environment:dict=None):
        if local_environment:
            local_initialization_policy(local_environment)
            self.local_environment = local_environmnet
            self.remote_environment = remote_environment

    def initializing_local_model_registry(self):
        pass    

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(model_specification, module):
        return getattr(import_module(f'.fmlops_forecasters.tensorflow.{model_specification}'), module)

    def _instance_basis(self, train_specification):
        pass

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_ailever_source_repository(self):
        return None

    def load_from_local_rawdata_repository(self):
        return None

    def load_from_local_feature_store(self):
        return None

    def load_from_local_source_repository(self):
        return None

    def load_from_local_model_registry(self):
        return None

    def load_from_local_metadata_store(self):
        return None

    def remote_from_local_rawdata_repository(self):
        return None

    def remote_from_local_feature_store(self):
        return None

    def remote_from_remote_source_repository(self):
        return None

    def remote_from_remote_model_registry(self):
        return None

    def remote_from_remote_matadata_store(self):
        return None

    def save_in_local_feature_store(self):
        pass

    def save_in_local_source_repository(self):
        pass

    def save_in_local_model_registry(self):
        pass

    def save_in_local_metadata_store(self):
        pass

    def save_in_remote_feature_store(self):
        pass

    def save_in_remote_source_repository(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def save_in_remote_metadata_store(self):
        pass

    def ModelTransferCore(self):
        modelcore = ModelTransferCore()
        return modelcore



class SklearnTriggerBlock(BaseTriggerBlock):
    def __init__(self, training_info:dict, local_environment:dict=None, remote_environment:dict=None):
        if local_environment:
            local_initialization_policy(local_environment)
            self.local_environment = local_environmnet
            self.remote_environment = remote_environment

    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(model_specification, module):
        return getattr(import_module(f'.fmlops_forecasters.sklearn.{model_specification}'), module)

    def _instance_basis(self, train_specification):
        pass

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_ailever_source_repository(self):
        return None

    def load_from_local_rawdata_repository(self):
        return None

    def load_from_local_feature_store(self):
        return None

    def load_from_local_source_repository(self):
        return None

    def load_from_local_model_registry(self):
        return None

    def load_from_local_metadata_store(self):
        return None

    def remote_from_local_rawdata_repository(self):
        return None

    def remote_from_local_feature_store(self):
        return None

    def remote_from_remote_source_repository(self):
        return None

    def remote_from_remote_model_registry(self):
        return None

    def remote_from_remote_matadata_store(self):
        return None

    def save_in_local_feature_store(self):
        pass

    def save_in_local_source_repository(self):
        pass

    def save_in_local_model_registry(self):
        pass

    def save_in_local_metadata_store(self):
        pass

    def save_in_remote_feature_store(self):
        pass

    def save_in_remote_source_repository(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def save_in_remote_metadata_store(self):
        pass

    def ModelTransferCore(self):
        modelcore = ModelTransferCore()
        return modelcore

class StatsmodelsTriggerBlock(BaseTriggerBlock):
    def __init__(self, training_info:dict, local_environment:dict=None, remote_environment:dict=None):
        if local_environment:
            local_initialization_policy(local_environment)
            self.local_environment = local_environmnet
            self.remote_environment = remote_environment

    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(model_specification, module):
        return getattr(import_module(f'.fmlops_forecasters.statsmodels.{model_specification}'), module)

    def _instance_basis(self, source):
        pass

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_ailever_source_repository(self):
        return None

    def load_from_local_rawdata_repository(self):
        return None

    def load_from_local_feature_store(self):
        return None

    def load_from_local_source_repository(self):
        return None

    def load_from_local_model_registry(self):
        return None

    def load_from_local_metadata_store(self):
        return None

    def remote_from_local_rawdata_repository(self):
        return None

    def remote_from_local_feature_store(self):
        return None

    def remote_from_remote_source_repository(self):
        return None

    def remote_from_remote_model_registry(self):
        return None

    def remote_from_remote_matadata_store(self):
        return None

    def save_in_local_feature_store(self):
        pass

    def save_in_local_source_repository(self):
        pass

    def save_in_local_model_registry(self):
        pass

    def save_in_local_metadata_store(self):
        pass

    def save_in_remote_feature_store(self):
        pass

    def save_in_remote_source_repository(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def save_in_remote_metadata_store(self):
        pass

    def ModelTransferCore(self):
        modelcore = ModelTransferCore()
        return modelcore


