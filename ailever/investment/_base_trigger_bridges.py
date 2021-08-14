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
base_dir['model_specification'] = fmlops_bs.local_system.root.metadata_store.model_specification.name

dir_path = dict()
dir_path['forecasting_model_registry'] = fmlops_bs.local_system.root.model_registry.forecasting_model_registry.path



class TorchTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(architecture:str, module:str):
        return getattr(import_module(f'{architecture}'), module)

    def _instance_basis(self, specification:dict, usage:str='train'):
        architecture = specification['architecture']
        InvestmentDataLoader = self._dynamic_import(architecture, 'InvestmentDataLoader')
        Model = self._dynamic_import(architecture, 'Model')
        Criterion = self._dynamic_import(architecture, 'Criterion')
        Optimizer = self._dynamic_import(architecture, 'Optimizer')
        retrainable_conditions = self._dynamic_import(architecture, 'retrainable_conditions')

        train_dataloader, test_dataloader = InvestmentDataLoader(specification)
        if specification['device'] == 'cpu':
            model = Model(specification)
            criterion = Criterion(specification)
        elif specification['device'] == 'cuda':
            model = Model(specification).cuda()
            criterion = Criterion(specification).cuda()
        optimizer = Optimizer(model, specification)
        
        return train_dataloader, test_dataloader, model, criterion, optimizer

    def instance_basis(self, specification:dict, usage:str='train'):
        return self.core_instances.pop('train_dataloader'), self.core_instances.pop('test_dataloader'), self.core_instances.pop('model'), self.core_instances.pop('criterion'), self.core_instances.pop('optimizer')

    def load_from_ailever_feature_store(self, specification:dict, usage:str='train'):
        self.core_instances = dict()

    def load_from_ailever_source_repository(self, specification:dict, usage:str='train'):
        train_dataloader, test_dataloader, model, criterion, optimizer = self._instance_basis(specification, usage)
        self.core_instances['train_dataloader'] = train_dataloader
        self.core_instances['test_dataloader'] = test_dataloader
        self.core_instances['model'] = model
        self.core_instances['criterion'] = criterion
        self.core_instances['optimizer'] = optimizer

    def load_from_ailever_model_registry(self, specification:dict, usage:str='train'):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, specification:dict, usage:str='train'):
        # return : model_specification, outcome_reports
        return None

    def load_from_local_feature_store(self, specification:dict, usage:str='train'):
        self.core_instances = dict()

    def load_from_local_source_repository(self, specification:dict, usage:str='train'):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, specification:dict, usage:str='train'):
        source = specification['loading_model_name_from_local_model_registry']
        if source:
            checkpoint = torch.load(os.path.join(dir_path['forecasting_model_registry'], source))
            self.core_instances['model'].load_state_dict(checkpoint['model_state_dict'])
            self.core_instances['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])

    def load_from_local_metadata_store(self, specification:dict, usage:str='train'):
        # return : model_specification, outcome_reports
        return None


    def load_from_remote_feature_store(self, specification:dict, usage:str='train'):
        self.core_instances = dict()

    def load_from_remote_source_repository(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_model_registry(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_metadata_store(self, specification:dict, usage:str='train'):
        return None


    def save_in_ailever_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_metadata_store(self, specification:dict, usage:str='train'):
        pass


    def save_in_local_feature_store(self, specification:dict, usage:str='train'):
        # [-]
        pass

    def save_in_local_source_repository(self, specification:dict, usage:str='train'):
        # [-]
        pass

    def save_in_local_model_registry(self, specification:dict, usage:str='train'):
        # [-]
        saving_path = os.path.join(dir_path['forecasting_model_registry'], specification['saving_name_in_local_model_registry']+'.pt')
        print(f"* Model's informations is saved({saving_path}).")
        torch.save({
            'model_state_dict': self.registry['model'].to('cpu').state_dict(),
            'optimizer_state_dict': self.registry['optimizer'].state_dict(),
            'epochs' : self.registry['epochs'],
            'cumulative_epochs' : self.registry['cumulative_epochs'],
            'training_loss': self.registry['train_mse'],
            'validation_loss': self.registry['validation_mse']}, saving_path)

    def save_in_local_metadata_store(self, specification:dict, usage:str='train'):
        pass


    def save_in_remote_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_metadata_store(self, specification:dict, usage:str='train'):
        pass



class TensorflowTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(model_specification:str, module:str):
        return getattr(import_module(f'.fmlops_forecasters.tensorflow.{model_specification}'), module)

    def _instance_basis(self, specification:dict, usage:str='train'):
        return None

    def instance_basis(self, specification:dict, usage:str='train'):
        return None

    def load_from_ailever_feature_store(self, specification:dict, usage:str='train'):
        # return : features
        return None

    def load_from_ailever_source_repository(self, specification:dict, usage:str='train'):
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, specification:dict, usage:str='train'):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, specification:dict, usage:str='train'):
        # return : model_specification, outcome_reports
        return None


    def load_from_local_feature_store(self, specification:dict, usage:str='train'):
        # return : features
        return None

    def load_from_local_source_repository(self, specification:dict, usage:str='train'):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, specification:dict, usage:str='train'):
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, specification:dict, usage:str='train'):
        # return : model_specification, outcome_reports
        return None


    def load_from_remote_feature_store(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_source_repository(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_model_registry(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_metadata_store(self, specification:dict, usage:str='train'):
        return None


    def save_in_ailever_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_metadata_store(self, specification:dict, usage:str='train'):
        pass


    def save_in_local_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_local_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_local_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_local_metadata_store(self, specification:dict, usage:str='train'):
        pass


    def save_in_remote_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_metadata_store(self, specification:dict, usage:str='train'):
        pass



class SklearnTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(model_specification:str, module:str):
        return getattr(import_module(f'.fmlops_forecasters.sklearn.{model_specification}'), module)

    def _instance_basis(self, specification:dict, usage:str='train'):
        return None

    def instance_basis(self, specification:dict, usage:str='train'):
        return None

    def load_from_ailever_feature_store(self, specification:dict, usage:str='train'):
        # return : features
        return None

    def load_from_ailever_source_repository(self, specification:dict, usage:str='train'):
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, specification:dict, usage:str='train'):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, specification:dict, usage:str='train'):
        # return : model_specification, outcome_reports
        return None


    def load_from_local_feature_store(self, specification:dict, usage:str='train'):
        # return : features
        return None

    def load_from_local_source_repository(self, specification:dict, usage:str='train'):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, specification:dict, usage:str='train'):
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, specification:dict, usage:str='train'):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_source_repository(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_model_registry(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_metadata_store(self, specification:dict, usage:str='train'):
        return None


    def save_in_ailever_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_metadata_store(self, specification:dict, usage:str='train'):
        pass


    def save_in_local_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_local_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_local_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_local_metadata_store(self, specification:dict, usage:str='train'):
        pass


    def save_in_remote_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_metadata_store(self, specification:dict, usage:str='train'):
        pass



class StatsmodelsTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(model_specification:str, module:str):
        return getattr(import_module(f'.fmlops_forecasters.statsmodels.{model_specification}'), module)

    def _instance_basis(self, specification:dict, usage:str='train'):
        return None

    def instance_basis(self, specification:dict, usage:str='train'):
        return None

    def load_from_ailever_feature_store(self, specification:dict, usage:str='train'):
        # return : features
        return None

    def load_from_ailever_source_repository(self, specification:dict, usage:str='train'):
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, specification:dict, usage:str='train'):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, specification:dict, usage:str='train'):
        # return : model_specification, outcome_reports
        return None


    def load_from_local_feature_store(self, specification:dict, usage:str='train'):
        # return : features
        return None

    def load_from_local_source_repository(self, specification:dict, usage:str='train'):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, specification:dict, usage:str='train'):
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, specification:dict, usage:str='train'):
        # return : model_specification, outcome_reports
        return None


    def load_from_remote_feature_store(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_source_repository(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_model_registry(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_metadata_store(self, specification:dict, usage:str='train'):
        return None


    def save_in_ailever_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_metadata_store(self, specification:dict, usage:str='train'):
        pass


    def save_in_local_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_local_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_local_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_local_metadata_store(self, specification:dict, usage:str='train'):
        pass


    def save_in_remote_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_metadata_store(self, specification:dict, usage:str='train'):
        pass
