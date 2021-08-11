from ailever.investment import __fmlops_bs__ as fmlops_bs
from .__base_structures import BaseTriggerBridge

import os
from importlib import import_module
import torch

__all__ = ['TorchTriggerBridge', 'TensorflowTriggerBridge', 'SklearnTriggerBridge', 'StatsmodelsTriggerBridge']


base_dir = dict()
base_dir['root'] = fmlops_bs.local_system.root.name
base_dir['feature_store'] = fmlops_bs.local_system.root.feature_store.name
base_dir['model_registry'] = fmlops_bs.local_system.root.model_registry.name
base_dir['source_repotitory'] = fmlops_bs.local_system.root.source_repository.name
base_dir['metadata_store'] = fmlops_bs.local_system.root.metadata_store.name
base_dir['model_specifications'] = fmlops_bs.local_system.root.metadata_store.model_specifications.name

dir_path = dict()
dir_path['model_registry'] = fmlops_bs.local_system.root.model_registry.path
dir_path['model_specifications'] = fmlops_bs.local_system.root.metadata_store.model_specifications.path



class TorchTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(architecture:str, module:str):
        return getattr(import_module(f'{architecture}'), module)

    def _instance_basis(self, train_specification:dict):
        architecture = train_specification['architecture']
        InvestmentDataLoader = self._dynamic_import(architecture, 'InvestmentDataLoader')
        Model = self._dynamic_import(architecture, 'Model')
        Criterion = self._dynamic_import(architecture, 'Criterion')
        Optimizer = self._dynamic_import(architecture, 'Optimizer')
        retrainable_conditions = self._dynamic_import(architecture, 'retrainable_conditions')

        train_dataloader, test_dataloader = InvestmentDataLoader(train_specification)
        if train_specification['device'] == 'cpu':
            model = Model(train_specification)
            criterion = Criterion(train_specification)
        elif train_specification['device'] == 'cuda':
            model = Model(train_specification).cuda()
            criterion = Criterion(train_specification).cuda()
        optimizer = Optimizer(model, train_specification)
        
        return train_dataloader, test_dataloader, model, criterion, optimizer

    def instance_basis(self, train_specification:dict):
        return self.core_instances.pop('train_dataloader'), self.core_instances.pop('test_dataloader'), self.core_instances.pop('model'), self.core_instances.pop('criterion'), self.core_instances.pop('optimizer')

    def load_from_ailever_feature_store(self, train_specification:dict):
        self.core_instances = dict()

    def load_from_ailever_source_repository(self, train_specification:dict):
        train_dataloader, test_dataloader, model, criterion, optimizer = self._instance_basis(train_specification)
        self.core_instances['train_dataloader'] = train_dataloader
        self.core_instances['test_dataloader'] = test_dataloader
        self.core_instances['model'] = model
        self.core_instances['criterion'] = criterion
        self.core_instances['optimizer'] = optimizer

    def load_from_ailever_model_registry(self, train_specification:dict):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, train_specification:dict):
        # return : model_specifications, outcome_reports
        return None

    def load_from_local_feature_store(self, train_specification:dict):
        self.core_instances = dict()

    def load_from_local_source_repository(self, train_specification:dict):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, train_specification:dict):
        source = train_specification['loading_model_name_from_local_model_registry']
        if source:
            checkpoint = torch.load(os.path.join(dir_path['model_registry'], source))
            self.core_instances['model'].load_state_dict(checkpoint['model_state_dict'])
            self.core_instances['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])

    def load_from_local_metadata_store(self, train_specification:dict):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, train_specification:dict):
        self.core_instances = dict()

    def load_from_remote_source_repository(self, train_specification:dict):
        return None

    def load_from_remote_model_registry(self, train_specification:dict):
        return None

    def load_from_remote_metadata_store(self, train_specification:dict):
        return None


    def save_in_ailever_feature_store(self, train_specification:dict):
        pass

    def save_in_ailever_source_repository(self, train_specification:dict):
        pass

    def save_in_ailever_model_registry(self, train_specification:dict):
        pass

    def save_in_ailever_metadata_store(self, train_specification:dict):
        pass


    def save_in_local_feature_store(self, train_specification:dict):
        # [-]
        pass

    def save_in_local_source_repository(self, train_specification:dict):
        # [-]
        pass

    def save_in_local_model_registry(self, train_specification:dict):
        # [-]
        saving_path = os.path.join(dir_path['model_registry'], train_specification['saving_name_in_local_model_registry']+'.pt')
        print(f"* Model's informations is saved({saving_path}).")
        torch.save({
            'model_state_dict': self.registry['model'].to('cpu').state_dict(),
            'optimizer_state_dict': self.registry['optimizer'].state_dict(),
            'epochs' : self.registry['epochs'],
            'cumulative_epochs' : self.registry['cumulative_epochs'],
            'training_loss': self.registry['train_mse'],
            'validation_loss': self.registry['validation_mse']}, saving_path)

    def save_in_local_metadata_store(self, train_specification:dict):
        pass


    def save_in_remote_feature_store(self, train_specification:dict):
        pass

    def save_in_remote_source_repository(self, train_specification:dict):
        pass

    def save_in_remote_model_registry(self, train_specification:dict):
        pass

    def save_in_remote_metadata_store(self, train_specification:dict):
        pass



class TensorflowTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(model_specification:str, module:str):
        return getattr(import_module(f'.fmlops_forecasters.tensorflow.{model_specification}'), module)

    def _instance_basis(self, train_specification:dict):
        return None

    def instance_basis(self, train_specification:dict):
        return None

    def load_from_ailever_feature_store(self, train_specification:dict):
        # return : features
        return None

    def load_from_ailever_source_repository(self, train_specification:dict):
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, train_specification:dict):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, train_specification:dict):
        # return : model_specifications, outcome_reports
        return None


    def load_from_local_feature_store(self, train_specification:dict):
        # return : features
        return None

    def load_from_local_source_repository(self, train_specification:dict):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, train_specification:dict):
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, train_specification:dict):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, train_specification:dict):
        return None

    def load_from_remote_source_repository(self, train_specification:dict):
        return None

    def load_from_remote_model_registry(self, train_specification:dict):
        return None

    def load_from_remote_metadata_store(self, train_specification:dict):
        return None


    def save_in_ailever_feature_store(self, train_specification:dict):
        pass

    def save_in_ailever_source_repository(self, train_specification:dict):
        pass

    def save_in_ailever_model_registry(self, train_specification:dict):
        pass

    def save_in_ailever_metadata_store(self, train_specification:dict):
        pass


    def save_in_local_feature_store(self, train_specification:dict):
        pass

    def save_in_local_source_repository(self, train_specification:dict):
        pass

    def save_in_local_model_registry(self, train_specification:dict):
        pass

    def save_in_local_metadata_store(self, train_specification:dict):
        pass


    def save_in_remote_feature_store(self, train_specification:dict):
        pass

    def save_in_remote_source_repository(self, train_specification:dict):
        pass

    def save_in_remote_model_registry(self, train_specification:dict):
        pass

    def save_in_remote_metadata_store(self, train_specification:dict):
        pass



class SklearnTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(model_specification:str, module:str):
        return getattr(import_module(f'.fmlops_forecasters.sklearn.{model_specification}'), module)

    def _instance_basis(self, train_specification:dict):
        return None

    def instance_basis(self, train_specification:dict):
        return None

    def load_from_ailever_feature_store(self, train_specification:dict):
        # return : features
        return None

    def load_from_ailever_source_repository(self, train_specification:dict):
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, train_specification:dict):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, train_specification:dict):
        # return : model_specifications, outcome_reports
        return None


    def load_from_local_feature_store(self, train_specification:dict):
        # return : features
        return None

    def load_from_local_source_repository(self, train_specification:dict):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, train_specification:dict):
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, train_specification:dict):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, train_specification:dict):
        return None

    def load_from_remote_source_repository(self, train_specification:dict):
        return None

    def load_from_remote_model_registry(self, train_specification:dict):
        return None

    def load_from_remote_metadata_store(self, train_specification:dict):
        return None


    def save_in_ailever_feature_store(self, train_specification:dict):
        pass

    def save_in_ailever_source_repository(self, train_specification:dict):
        pass

    def save_in_ailever_model_registry(self, train_specification:dict):
        pass

    def save_in_ailever_metadata_store(self, train_specification:dict):
        pass


    def save_in_local_feature_store(self, train_specification:dict):
        pass

    def save_in_local_source_repository(self, train_specification:dict):
        pass

    def save_in_local_model_registry(self, train_specification:dict):
        pass

    def save_in_local_metadata_store(self, train_specification:dict):
        pass


    def save_in_remote_feature_store(self, train_specification:dict):
        pass

    def save_in_remote_source_repository(self, train_specification:dict):
        pass

    def save_in_remote_model_registry(self, train_specification:dict):
        pass

    def save_in_remote_metadata_store(self, train_specification:dict):
        pass



class StatsmodelsTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(model_specification:str, module:str):
        return getattr(import_module(f'.fmlops_forecasters.statsmodels.{model_specification}'), module)

    def _instance_basis(self, train_specification:dict):
        return None

    def instance_basis(self, train_specification:dict):
        return None

    def load_from_ailever_feature_store(self, train_specification:dict):
        # return : features
        return None

    def load_from_ailever_source_repository(self, train_specification:dict):
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, train_specification:dict):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, train_specification:dict):
        # return : model_specifications, outcome_reports
        return None


    def load_from_local_feature_store(self, train_specification:dict):
        # return : features
        return None

    def load_from_local_source_repository(self, train_specification:dict):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, train_specification:dict):
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, train_specification:dict):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, train_specification:dict):
        return None

    def load_from_remote_source_repository(self, train_specification:dict):
        return None

    def load_from_remote_model_registry(self, train_specification:dict):
        return None

    def load_from_remote_metadata_store(self, train_specification:dict):
        return None


    def save_in_ailever_feature_store(self, train_specification:dict):
        pass

    def save_in_ailever_source_repository(self, train_specification:dict):
        pass

    def save_in_ailever_model_registry(self, train_specification:dict):
        pass

    def save_in_ailever_metadata_store(self, train_specification:dict):
        pass


    def save_in_local_feature_store(self, train_specification:dict):
        pass

    def save_in_local_source_repository(self, train_specification:dict):
        pass

    def save_in_local_model_registry(self, train_specification:dict):
        pass

    def save_in_local_metadata_store(self, train_specification:dict):
        pass


    def save_in_remote_feature_store(self, train_specification:dict):
        pass

    def save_in_remote_source_repository(self, train_specification:dict):
        pass

    def save_in_remote_model_registry(self, train_specification:dict):
        pass

    def save_in_remote_metadata_store(self, train_specification:dict):
        pass



