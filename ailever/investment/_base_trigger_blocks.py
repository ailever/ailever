from ailever.investment import fmlops_bs
from .__base_structures import BaseTriggerBlock
from ._fmlops_policy import local_initialization_policy, remote_initialization_policy
from ._base_transfer import ModelTransferCore

import sys
import os
from importlib import import_module
from functools import partial
import torch

base_dir = dict()
base_dir['root'] = fmlops_bs.local_system.root.name
base_dir['rawdata_repository'] = fmlops_bs.local_system.root.rawdata_repository.name
base_dir['feature_store'] = fmlops_bs.local_system.root.feature_store.name
base_dir['model_registry'] = fmlops_bs.local_system.root.model_registry.name
base_dir['source_repotitory'] = fmlops_bs.local_system.root.source_repository.name
base_dir['metadata_store'] = fmlops_bs.local_system.root.metadata_store.name
base_dir['model_specifications'] = fmlops_bs.local_system.root.metadata_store.model_specifications.name

dir_path = dict()
dir_path['model_registry'] = fmlops_bs.local_system.root.model_registry.path
dir_path['model_specifications'] = fmlops_bs.local_system.root.metadata_store.model_specifications.path

class TorchTriggerBlock(BaseTriggerBlock):
    def __init__(self, local_environment:dict=None, remote_environment:dict=None):
        self.registry = dict()
        sys.path.append(os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fmlops_forecasters'), 'torch'))

        if local_environment:
            self.local_environment = local_environment
            local_initialization_policy(self.local_environment)

        if remote_environment:
            self.remote_environment = remote_environment
            remote_initialization_policy(self.remote_environment)

    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass
    
    @staticmethod
    def _dynamic_import(architecture, module):
        return getattr(import_module(f'{architecture}'), module)

    def _instance_basis(self, train_specification):
        architecture = train_specification['architecture']
        InvestmentDataLoader = self._dynamic_import(architecture, 'InvestmentDataLoader')
        Model = self._dynamic_import(architecture, 'Model')
        Criterion = self._dynamic_import(architecture, 'Criterion')
        Optimizer = self._dynamic_import(architecture, 'Optimizer')

        train_dataloader, test_dataloader = InvestmentDataLoader(train_specification)
        if train_specification['device'] == 'cpu':
            model = Model(train_specification)
            criterion = Criterion(train_specification)
        elif train_specification['device'] == 'cuda':
            model = Model(train_specification).cuda()
            criterion = Criterion(train_specification).cuda()
        optimizer = Optimizer(model, train_specification)

        # instance update
        """
        checkpoint = torch.load(os.path.join(source_repository['source_repository'], source))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        """
        return train_dataloader, test_dataloader, model, criterion, optimizer

    def train(self, train_specification):
        epochs = train_specification['epochs']
        device = train_specification['device']
        train_dataloader, test_dataloader, model, criterion, optimizer = self._instance_basis(train_specification)

        for epoch in range(epochs):
            training_losses = []
            model.train()
            for batch_idx, (x_train, y_train) in enumerate(train_dataloader):
                # Forward
                hypothesis = model(x_train.squeeze().float().to(device))
                cost = criterion(hypothesis.squeeze().float().to(device), y_train.squeeze().float().to(device))
                # Backward
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                training_losses.append(cost)
            # Alert
            TrainMSE = torch.mean(torch.tensor(training_losses)).data
            if 'cumulative_epochs' in train_specification.keys():
                previous_cumulative_epochs = train_specification['cumulative_epochs']
                cumulative_epochs = train_specification['cumulative_epochs'] + epochs
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
                    hypothesis = model(x_train.squeeze().float().to(device))
                    cost = criterion(hypothesis.squeeze().float().to(device), y_train.squeeze().float().to(device))
                    validation_losses.append(cost)
                # Alert
                ValidationMSE = torch.mean(torch.tensor(validation_losses)).data
                if 'cumulative_epochs' in train_specification.keys():
                    previous_cumulative_epochs = train_specification['cumulative_epochs']
                    cumulative_epochs = train_specification['cumulative_epochs'] + epochs
                    print(f'[Validation][{epoch+1+previous_cumulative_epochs}/{cumulative_epochs}]', float(ValidationMSE))
                else:
                    print(f'[Validation][{epoch+1}/{epochs}]', float(ValidationMSE))

        self.registry['model'] = model
        self.registry['optimizer'] = optimizer
        self.registry['cumulative_epochs'] = train_specification['cumulative_epochs']+epochs if 'cumulative_epochs' in train_specification.keys() else epochs
        self.registry['train_mse'] = TrainMSE
        self.registry['validation_mse'] = ValidationMSE

    def prediction(self):
        pass

    def outcome_report(self):
        pass

    def load_from_ailever_source_repository(self):
        return None

    def load_from_local_rawdata_repository(self):
        return None

    def load_from_local_feature_store(self):
        return None

    def load_from_local_source_repository(self):
        return None

    def load_from_local_model_registry(self, train_specification):
        return None

    def load_from_local_metadata_store(self):
        return None

    def load_from_remote_rawdata_repository(self):
        return None

    def load_from_remote_feature_store(self):
        return None

    def load_from_remote_source_repository(self):
        return None

    def load_from_remote_model_registry(self):
        return None

    def load_from_remote_metadata_store(self):
        return None

    def save_in_local_feature_store(self):
        pass

    def save_in_local_source_repository(self):
        pass

    def save_in_local_model_registry(self):
        print(f"* Model's informations is saved({dir_path['model_specifications']}).")
        torch.save({
            'model_state_dict': self.registry['model'].to('cpu').state_dict(),
            'optimizer_state_dict': self.registry['optimizer'].state_dict(),
            'epochs' : self.registry['epochs'],
            'cumulative_epochs' : self.registry['cumulative_epochs'],
            'training_loss': self.registry['train_mse'],
            'validation_loss': self.registry['validation_mse']}, dir_path['model_specifications'])

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
        sys.path.append(os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fmlops_forecasters'), 'tensorflow'))
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
        sys.path.append(os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fmlops_forecasters'), 'sklearn'))
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
        sys.path.append(os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fmlops_forecasters'), 'statsmodels'))
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


