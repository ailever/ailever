from ailever.investment import fmlops_bs
from .__base_structures import BaseTriggerBridge

import os
from importlib import import_module
import torch

__all__ = ['TorchTriggerBridge', 'TensorflowTriggerBridge', 'SklearnTriggerBridge', 'StatsmodelsTriggerBridge']

class TorchTriggerBridge(BaseTriggerBridge):
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

    def load_from_ailever_feature_store(self, train_specification):
        # [-]
        # return : features
        return None

    def load_from_ailever_source_repository(self, train_specification):
        # [-]
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_local_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_local_source_repository(self, train_specification):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, train_specification):
        # [-]
        # instance update
        """
        checkpoint = torch.load(os.path.join(source_repository['source_repository'], source))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        """
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, train_specification):
        return None

    def load_from_remote_source_repository(self, train_specification):
        return None

    def load_from_remote_model_registry(self, train_specification):
        return None

    def load_from_remote_metadata_store(self, train_specification):
        return None


    def save_in_ailever_feature_store(self, train_specification):
        pass

    def save_in_ailever_source_repository(self, train_specification):
        pass

    def save_in_ailever_model_registry(self, train_specification):
        pass

    def save_in_ailever_metadata_store(self, train_specification):
        pass


    def save_in_local_feature_store(self, train_specification):
        # [-]
        pass

    def save_in_local_source_repository(self, train_specification):
        # [-]
        pass

    def save_in_local_model_registry(self, train_specification):
        # [-]
        saving_path = os.path.join(dir_path['model_specifications'], train_specification['saving_name']+'.pt')
        print(f"* Model's informations is saved({saving_path}).")
        torch.save({
            'model_state_dict': self.registry['model'].to('cpu').state_dict(),
            'optimizer_state_dict': self.registry['optimizer'].state_dict(),
            'epochs' : self.registry['epochs'],
            'cumulative_epochs' : self.registry['cumulative_epochs'],
            'training_loss': self.registry['train_mse'],
            'validation_loss': self.registry['validation_mse']}, saving_path)

    def save_in_local_metadata_store(self, train_specification):
        pass


    def save_in_remote_feature_store(self, train_specification):
        pass

    def save_in_remote_source_repository(self, train_specification):
        pass

    def save_in_remote_model_registry(self, train_specification):
        pass

    def save_in_remote_metadata_store(self, train_specification):
        pass



class TensorflowTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(model_specification, module):
        return getattr(import_module(f'.fmlops_forecasters.tensorflow.{model_specification}'), module)

    def _instance_basis(self, train_specification):
        pass

    def load_from_ailever_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_ailever_source_repository(self, train_specification):
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_local_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_local_source_repository(self, train_specification):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, train_specification):
        return None

    def load_from_remote_source_repository(self, train_specification):
        return None

    def load_from_remote_model_registry(self, train_specification):
        return None

    def load_from_remote_metadata_store(self, train_specification):
        return None


    def save_in_ailever_feature_store(self, train_specification):
        pass

    def save_in_ailever_source_repository(self, train_specification):
        pass

    def save_in_ailever_model_registry(self, train_specification):
        pass

    def save_in_ailever_metadata_store(self, train_specification):
        pass


    def save_in_local_feature_store(self, train_specification):
        pass

    def save_in_local_source_repository(self, train_specification):
        pass

    def save_in_local_model_registry(self, train_specification):
        pass

    def save_in_local_metadata_store(self, train_specification):
        pass


    def save_in_remote_feature_store(self, train_specification):
        pass

    def save_in_remote_source_repository(self, train_specification):
        pass

    def save_in_remote_model_registry(self, train_specification):
        pass

    def save_in_remote_metadata_store(self, train_specification):
        pass



class SklearnTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(model_specification, module):
        return getattr(import_module(f'.fmlops_forecasters.sklearn.{model_specification}'), module)

    def _instance_basis(self, train_specification):
        pass

    def load_from_ailever_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_ailever_source_repository(self, train_specification):
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_local_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_local_source_repository(self, train_specification):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, train_specification):
        return None

    def load_from_remote_source_repository(self, train_specification):
        return None

    def load_from_remote_model_registry(self, train_specification):
        return None

    def load_from_remote_metadata_store(self, train_specification):
        return None


    def save_in_ailever_feature_store(self, train_specification):
        pass

    def save_in_ailever_source_repository(self, train_specification):
        pass

    def save_in_ailever_model_registry(self, train_specification):
        pass

    def save_in_ailever_metadata_store(self, train_specification):
        pass


    def save_in_local_feature_store(self, train_specification):
        pass

    def save_in_local_source_repository(self, train_specification):
        pass

    def save_in_local_model_registry(self, train_specification):
        pass

    def save_in_local_metadata_store(self, train_specification):
        pass


    def save_in_remote_feature_store(self, train_specification):
        pass

    def save_in_remote_source_repository(self, train_specification):
        pass

    def save_in_remote_model_registry(self, train_specification):
        pass

    def save_in_remote_metadata_store(self, train_specification):
        pass



class StatsmodelsTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    @staticmethod
    def _dynamic_import(model_specification, module):
        return getattr(import_module(f'.fmlops_forecasters.statsmodels.{model_specification}'), module)

    def _instance_basis(self, source):
        pass

    def load_from_ailever_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_ailever_source_repository(self, train_specification):
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_local_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_local_source_repository(self, train_specification):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, train_specification):
        return None

    def load_from_remote_source_repository(self, train_specification):
        return None

    def load_from_remote_model_registry(self, train_specification):
        return None

    def load_from_remote_metadata_store(self, train_specification):
        return None


    def save_in_ailever_feature_store(self, train_specification):
        pass

    def save_in_ailever_source_repository(self, train_specification):
        pass

    def save_in_ailever_model_registry(self, train_specification):
        pass

    def save_in_ailever_metadata_store(self, train_specification):
        pass


    def save_in_local_feature_store(self, train_specification):
        pass

    def save_in_local_source_repository(self, train_specification):
        pass

    def save_in_local_model_registry(self, train_specification):
        pass

    def save_in_local_metadata_store(self, train_specification):
        pass


    def save_in_remote_feature_store(self, train_specification):
        pass

    def save_in_remote_source_repository(self, train_specification):
        pass

    def save_in_remote_model_registry(self, train_specification):
        pass

    def save_in_remote_metadata_store(self, train_specification):
        pass



