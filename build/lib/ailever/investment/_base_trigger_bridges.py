from ailever.investment import __fmlops_bs__ as fmlops_bs
from .__base_structures import BaseTriggerBridge

import os
from importlib import import_module
import torch

__all__ = ['TorchTriggerBridge', 'TensorflowTriggerBridge', 'SklearnTriggerBridge', 'StatsmodelsTriggerBridge']


base_dir_path = dict()
base_dir_path['forecasting_model_registry'] = fmlops_bs.local_system.root.model_registry.forecasting_model_registry.path
base_dir_path['fundamental_analysis_result'] = fmlops_bs.local_system.root.analysis_report_repository.fundamental_analysis_result.path
base_dir_path['technical_analysis_result'] = fmlops_bs.local_system.root.analysis_report_repository.technical_analysis_result.path
base_dir_path['model_prediction_result'] = fmlops_bs.local_system.root.analysis_report_repository.model_prediction_result.path
base_dir_path['sector_analysis_result'] = fmlops_bs.local_system.root.analysis_report_repository.sector_analysis_result.path


class TorchTriggerBridge(BaseTriggerBridge):
    def __init__(self):
        self.core_instances = dict()

    @staticmethod
    def _dynamic_import(architecture:str, module:str):
        return getattr(import_module(f'{architecture}'), module)

    def into_trigger_bridge(self, specification:dict, usage:str='train'):
        if usage == 'train':
            architecture = specification['architecture']
            InvestmentDataLoader = self._dynamic_import(architecture, 'InvestmentDataLoader')
            Model = self._dynamic_import(architecture, 'Model')
            Criterion = self._dynamic_import(architecture, 'Criterion')
            Optimizer = self._dynamic_import(architecture, 'Optimizer')
            retrainable_conditions = self._dynamic_import(architecture, 'retrainable_conditions')
            if specification['device'] == 'cpu':
                model = Model(specification)
                criterion = Criterion(specification)
            elif specification['device'] == 'cuda':
                model = Model(specification).cuda()
                criterion = Criterion(specification).cuda()
            optimizer = Optimizer(model, specification)
            train_dataloader, test_dataloader = InvestmentDataLoader(specification)
            return train_dataloader, test_dataloader, model, criterion, optimizer

        elif usage == 'prediction':
            architecture = specification['architecture']
            Scaler = self._dynamic_import(architecture, 'Scaler')
            InvestmentDataset = self._dynamic_import(architecture, 'InvestmentDataset')
            Model = self._dynamic_import(architecture, 'Model')
            retrainable_conditions = self._dynamic_import(architecture, 'retrainable_conditions')
            
            scaler = Scaler()
            model = Model(specification).to('cpu')
            investment_dataset = InvestmentDataset(specification).type(mode='test')
            return scaler, investment_dataset, model

    def into_trigger_block(self, specification:dict, usage:str='train'):
        if usage == 'train':
            return self.core_instances.pop('train_dataloader'), self.core_instances.pop('test_dataloader'), self.core_instances.pop('model'), self.core_instances.pop('criterion'), self.core_instances.pop('optimizer')
        elif usage == 'prediction':
            return self.core_instances.pop('investment_dataset'), self.core_instances.pop('model')

    def load_from_ailever_feature_store(self, specification:dict, usage:str='train'):
        pass

    def load_from_ailever_source_repository(self, specification:dict, usage:str='train'):
        if usage == 'train':
            train_dataloader, test_dataloader, model, criterion, optimizer = self.into_trigger_bridge(specification, usage)
            self.core_instances['train_dataloader'] = train_dataloader
            self.core_instances['test_dataloader'] = test_dataloader
            self.core_instances['model'] = model
            self.core_instances['criterion'] = criterion
            self.core_instances['optimizer'] = optimizer
        elif usage == 'prediction':
            scaler, investment_dataset, model = self.into_trigger_bridge(specification, usage)
            self.core_instances['investment_dataset'] = investment_dataset
            self.core_instances['model'] = model
            self.core_instances['scaler'] = scaler

    def load_from_ailever_model_registry(self, specification:dict, usage:str='train'):
        # return : model, optimizer
        return None

    def load_from_ailever_monitoring_source(self, specification:dict, usage:str='train'):
        pass

    def load_from_ailever_analysis_report_repository(self, specification:dict, usage:str='train'):
        pass

    def load_from_ailever_investment_outcome_repository(self, specification:dict, usage:str='train'):
        pass

    def load_from_ailever_metadata_store(self, specification:dict, usage:str='train'):
        # return : model_specification, outcome_reports
        return None

    def load_from_local_feature_store(self, specification:dict, usage:str='train'):
        pass

    def load_from_local_source_repository(self, specification:dict, usage:str='train'):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, specification:dict, usage:str='train'):
        if usage == 'train':
            source = specification['loading_model_name_from_local_model_registry']
            if source:
                checkpoint = torch.load(os.path.join(base_dir_path['forecasting_model_registry'], source))
                self.core_instances['model'].load_state_dict(checkpoint['model_state_dict'])
                self.core_instances['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])

        elif usage == 'prediction':
            source = specification['loading_model_name_from_local_model_registry']
            if source:
                checkpoint = torch.load(os.path.join(base_dir_path['forecasting_model_registry'], source))
                self.core_instances['model'].load_state_dict(checkpoint['model_state_dict'])
                
    def load_from_local_monitoring_source(self, specification:dict, usage:str='train'):
        pass

    def load_from_local_analysis_report_repository(self, specification:dict, usage:str='train'):
        pass

    def load_from_local_investment_outcome_repository(self, specification:dict, usage:str='train'):
        pass

    def load_from_local_metadata_store(self, specification:dict, usage:str='train'):
        # return : model_specification, outcome_reports
        return None


    def load_from_remote_feature_store(self, specification:dict, usage:str='train'):
        pass

    def load_from_remote_source_repository(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_model_registry(self, specification:dict, usage:str='train'):
        return None

    def load_from_remote_analysis_report_repository(self, specification:dict, usage:str='train'):
        pass

    def load_from_remote_monitoring_source(self, specification:dict, usage:str='train'):
        pass

    def load_from_remote_investment_outcome_repository(self, specification:dict, usage:str='train'):
        pass

    def load_from_remote_metadata_store(self, specification:dict, usage:str='train'):
        return None


    def save_in_ailever_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_analysis_report_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_monitoring_source(self, specification:dict, usage:str='train'):
        pass

    def save_in_ailever_investment_outcome_repository(self, specification:dict, usage:str='train'):
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
        saving_path = os.path.join(specification['saving_path'], specification['saving_name_in_local_model_registry']+'.pt')
        print(f"* Model's informations is saved({saving_path}).")
        torch.save({
            'model_state_dict': self.forecasting_model_registry['model'].to('cpu').state_dict(),
            'optimizer_state_dict': self.forecasting_model_registry['optimizer'].state_dict(),
            'epochs' : self.forecasting_model_registry['epochs'],
            'cumulative_epochs' : self.forecasting_model_registry['cumulative_epochs'],
            'training_loss': self.forecasting_model_registry['train_mse'],
            'validation_loss': self.forecasting_model_registry['validation_mse']}, saving_path)

    def save_in_local_analysis_report_repository(self, specification:dict, usage:str='prediction'):
        if usage == 'prediction':
            prediction_table_file_name = os.path.join(base_dir_path['model_prediction_result'], specification['ticker'])
            self.analysis_report_repository['prediction_talbe'].to_csv(prediction_table_file_name+'.csv')
            for column, plt in self.analysis_report_repository['prediction_visualization'].items():
                prediction_image_file_name = os.path.join(base_dir_path['model_prediction_result'], specification['ticker'])
                plt.savefig( + f'_{column}.png')

    def save_in_local_monitoring_source(self, specification:dict, usage:str='train'):
        pass

    def save_in_local_investment_outcome_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_local_metadata_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_feature_store(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_source_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_model_registry(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_analysis_report_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_remotel_monitoring_source(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_investment_outcome_repository(self, specification:dict, usage:str='train'):
        pass

    def save_in_remote_metadata_store(self, specification:dict, usage:str='train'):
        pass



class TensorflowTriggerBridge(BaseTriggerBridge):
    def __init__(self):
        self.core_instances = dict()

    @staticmethod
    def _dynamic_import():
        pass
    
    def into_trigger_bridge(self):
        pass

    def into_trigger_block(self):
        pass

    def load_from_local_feature_store(self):
        pass

    def load_from_local_source_repository(self):
        pass

    def load_from_local_model_registry(self):
        pass

    def load_from_local_monitoring_source(self):
        pass

    def load_from_local_analysis_report_repository(self):
        pass

    def load_from_local_investment_outcome_repository(self):
        pass

    def load_from_local_metadata_store(self):
        pass

    def load_from_remote_feature_store(self):
        pass

    def load_from_remote_source_repository(self):
        pass

    def load_from_remote_model_registry(self):
        pass

    def load_from_remote_monitoring_source(self):
        pass

    def load_from_remote_analysis_report_repository(self):
        pass

    def load_from_remote_investment_outcome_repository(self):
        pass

    def load_from_remote_metadata_store(self):
        pass

    
    def save_in_local_feature_store(self):
        pass

    def save_in_local_source_repository(self):
        pass

    def save_in_local_model_registry(self):
        pass

    def save_from_local_monitoring_source(self):
        pass

    def save_from_local_analysis_report_repository(self):
        pass

    def save_from_local_investment_outcome_repository(self):
        pass

    
    def save_in_local_metadata_store(self):
        pass

    def save_in_remote_feature_store(self):
        pass

    def save_in_remote_source_repository(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def save_from_remote_monitoring_source(self):
        pass

    def save_from_remote_analysis_report_repository(self):
        pass

    def save_from_remote_investment_outcome_repository(self):
        pass

    def save_in_remote_metadata_store(self):
        pass


class SklearnTriggerBridge(BaseTriggerBridge):
    def __init__(self):
        self.core_instances = dict()

    @staticmethod
    def _dynamic_import():
        pass
    
    def into_trigger_bridge(self):
        pass

    def into_trigger_block(self):
        pass

    def load_from_local_feature_store(self):
        pass

    def load_from_local_source_repository(self):
        pass

    def load_from_local_model_registry(self):
        pass

    def load_from_local_monitoring_source(self):
        pass

    def load_from_local_analysis_report_repository(self):
        pass

    def load_from_local_investment_outcome_repository(self):
        pass

    def load_from_local_metadata_store(self):
        pass

    def load_from_remote_feature_store(self):
        pass

    def load_from_remote_source_repository(self):
        pass

    def load_from_remote_model_registry(self):
        pass

    def load_from_remote_monitoring_source(self):
        pass

    def load_from_remote_analysis_report_repository(self):
        pass

    def load_from_remote_investment_outcome_repository(self):
        pass

    def load_from_remote_metadata_store(self):
        pass

    
    def save_in_local_feature_store(self):
        pass

    def save_in_local_source_repository(self):
        pass

    def save_in_local_model_registry(self):
        pass

    def save_from_local_monitoring_source(self):
        pass

    def save_from_local_analysis_report_repository(self):
        pass

    def save_from_local_investment_outcome_repository(self):
        pass

    
    def save_in_local_metadata_store(self):
        pass

    def save_in_remote_feature_store(self):
        pass

    def save_in_remote_source_repository(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def save_from_remote_monitoring_source(self):
        pass

    def save_from_remote_analysis_report_repository(self):
        pass

    def save_from_remote_investment_outcome_repository(self):
        pass

    def save_in_remote_metadata_store(self):
        pass


class StatsmodelsTriggerBridge(BaseTriggerBridge):
    def __init__(self):
        self.core_instances = dict()

    @staticmethod
    def _dynamic_import():
        pass
    
    def into_trigger_bridge(self):
        pass

    def into_trigger_block(self):
        pass

    def load_from_local_feature_store(self):
        pass

    def load_from_local_source_repository(self):
        pass

    def load_from_local_model_registry(self):
        pass

    def load_from_local_monitoring_source(self):
        pass

    def load_from_local_analysis_report_repository(self):
        pass

    def load_from_local_investment_outcome_repository(self):
        pass

    def load_from_local_metadata_store(self):
        pass

    def load_from_remote_feature_store(self):
        pass

    def load_from_remote_source_repository(self):
        pass

    def load_from_remote_model_registry(self):
        pass

    def load_from_remote_monitoring_source(self):
        pass

    def load_from_remote_analysis_report_repository(self):
        pass

    def load_from_remote_investment_outcome_repository(self):
        pass

    def load_from_remote_metadata_store(self):
        pass

    
    def save_in_local_feature_store(self):
        pass

    def save_in_local_source_repository(self):
        pass

    def save_in_local_model_registry(self):
        pass

    def save_from_local_monitoring_source(self):
        pass

    def save_from_local_analysis_report_repository(self):
        pass

    def save_from_local_investment_outcome_repository(self):
        pass

    
    def save_in_local_metadata_store(self):
        pass

    def save_in_remote_feature_store(self):
        pass

    def save_in_remote_source_repository(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def save_from_remote_monitoring_source(self):
        pass

    def save_from_remote_analysis_report_repository(self):
        pass

    def save_from_remote_investment_outcome_repository(self):
        pass

    def save_in_remote_metadata_store(self):
        pass


