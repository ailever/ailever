from ailever.investment import __fmlops_bs__ as fmlops_bs
from .fmlops_management import FMR_Manager
from ._base_trigger_blocks import TorchTriggerBlock, TensorflowTriggerBlock, SklearnTriggerBlock, StatsmodelsTriggerBlock

import re
from pprint import pprint


class Forecaster:
    def __init__(self, local_environment:dict=None, remote_environment:dict=None):
        self.fmr_manager = FMR_Manager()
        
        self.trigger_block = dict()
        self.trigger_block['torch'] = TorchTriggerBlock(local_environment=local_environment, remote_environment=remote_environment)
        self.trigger_block['tensorflow'] = None
        self.trigger_block['xgboost'] = None
        self.trigger_block['sklearn'] = None
        self.trigger_block['statsmodels'] = None
 
    def integrated_trigger(self, baskets:list, integrated_specifications:dict):
        train_trigger(integrated_specifications['train'])
        prediction_trigger(integrated_specifications['prediction'])
        analysis_trigger(integrated_specifications['analysis'])
        evaluation_trigger(integrated_specifications['evaluation'])

    def train_trigger(self, baskets:list, train_specifications:dict):
        """
        * train_trigger loading process : [02]
        01] [X] ailever_feature_store > ailever_source_repository > ailever_model_registry 
        02] [O] ailever_feature_store > ailever_source_repository > local_model_registry
        03] [X] ailever_feature_store > ailever_source_repository > remote_model_registry
        04] [X] ailever_feature_store > local_source_repository > ailever_model_registry 
        05] [X] ailever_feature_store > local_source_repository > local_model_registry 
        06] [X] ailever_feature_store > local_source_repository > remote_model_registry 
        07] [X] ailever_feature_store > remote_source_repository > ailever_model_registry 
        08] [X] ailever_feature_store > remote_source_repository > local_model_registry 
        09] [X] ailever_feature_store > remote_source_repository > remote_model_registry 
        10] [X] local_feature_store > ailever_source_repository > ailever_model_registry 
        11] [X] local_feature_store > ailever_source_repository > local_model_registry
        12] [X] local_feature_store > ailever_source_repository > remote_model_registry
        13] [X] local_feature_store > local_source_repository > ailever_model_registry 
        14] [X] local_feature_store > local_source_repository > local_model_registry 
        15] [X] local_feature_store > local_source_repository > remote_model_registry 
        16] [X] local_feature_store > remote_source_repository > ailever_model_registry 
        17] [X] local_feature_store > remote_source_repository > local_model_registry 
        18] [X] local_feature_store > remote_source_repository > remote_model_registry 
        19] [X] remote_feature_store > ailever_source_repository > ailever_model_registry 
        20] [X] remote_feature_store > ailever_source_repository > local_model_registry
        21] [X] remote_feature_store > ailever_source_repository > remote_model_registry
        22] [X] remote_feature_store > local_source_repository > ailever_model_registry 
        23] [X] remote_feature_store > local_source_repository > local_model_registry 
        24] [X] remote_feature_store > local_source_repository > remote_model_registry 
        25] [X] remote_feature_store > remote_source_repository > ailever_model_registry 
        26] [X] remote_feature_store > remote_source_repository > local_model_registry 
        27] [X] remote_feature_store > remote_source_repository > remote_model_registry 

        * train_trigger storing process : [14]
        01] [X] ailever_feature_store > ailever_model_registry > ailever_metadata_store
        02] [X] ailever_feature_store > ailever_model_registry > local_metadata_store
        03] [X] ailever_feature_store > ailever_model_registry > remote_metadata_store
        04] [X] ailever_feature_store > local_model_registry > ailever_metadata_store
        05] [X] ailever_feature_store > local_model_registry > local_metadata_store
        06] [X] ailever_feature_store > local_model_registry > remote_metadata_store
        07] [X] ailever_feature_store > remote_model_registry > ailever_metadata_store
        08] [X] ailever_feature_store > remote_model_registry > local_metadata_store
        09] [X] ailever_feature_store > remote_model_registry > remote_metadata_store
        10] [X] local_feature_store > ailever_model_registry > ailever_metadata_store
        11] [X] local_feature_store > ailever_model_registry > local_metadata_store
        12] [X] local_feature_store > ailever_model_registry > remote_metadata_store
        13] [X] local_feature_store > local_model_registry > ailever_metadata_store
        14] [O] local_feature_store > local_model_registry > local_metadata_store
        15] [X] local_feature_store > local_model_registry > remote_metadata_store
        16] [X] local_feature_store > remote_model_registry > ailever_metadata_store
        17] [X] local_feature_store > remote_model_registry > local_metadata_store
        18] [X] local_feature_store > remote_model_registry > remote_metadata_store
        19] [X] remote_feature_store > ailever_model_registry > ailever_metadata_store
        20] [X] remote_feature_store > ailever_model_registry > local_metadata_store
        21] [X] remote_feature_store > ailever_model_registry > remote_metadata_store
        22] [X] remote_feature_store > local_model_registry > ailever_metadata_store
        23] [X] remote_feature_store > local_model_registry > local_metadata_store
        24] [X] remote_feature_store > local_model_registry > remote_metadata_store
        25] [X] remote_feature_store > remote_model_registry > ailever_metadata_store
        26] [X] remote_feature_store > remote_model_registry > local_metadata_store
        27] [X] remote_feature_store > remote_model_registry > remote_metadata_store
        """
        for security in baskets:
            # train_specification
            train_specifications[security]['ticker'] = security
            train_specification = train_specifications[security]
            framework = train_specification['framework']
            
            # initializing train_specification
            train_specification = self.trigger_block[framework].ui_buffer(train_specification, usage='train')
            
            # loading
            train_specification = self.fmr_manager.local_loading_connection(train_specification, usage='train')
            train_specification = self.fmr_manager.remote_loading_connection(train_specification, usage='train')
            self.trigger_block[framework].loaded_from(train_specification, usage='train')

            # training
            self.trigger_block[framework].train(train_specification)
            
            # storing
            train_specification = self.fmr_manager.local_storing_connection(train_specification, usage='train')
            train_specification = self.fmr_manager.remote_storing_connection(train_specification, usage='train')
            self.trigger_block[framework].store_in(train_specification, usage='train')
 
    def prediction_trigger(self, baskets:list, prediction_specifications:dict):
        """
        * prediction_trigger loading process : [02]
        01] [X] ailever_feature_store > ailever_source_repository > ailever_model_registry 
        02] [O] ailever_feature_store > ailever_source_repository > local_model_registry
        03] [X] ailever_feature_store > ailever_source_repository > remote_model_registry
        04] [X] ailever_feature_store > local_source_repository > ailever_model_registry 
        05] [X] ailever_feature_store > local_source_repository > local_model_registry 
        06] [X] ailever_feature_store > local_source_repository > remote_model_registry 
        07] [X] ailever_feature_store > remote_source_repository > ailever_model_registry 
        08] [X] ailever_feature_store > remote_source_repository > local_model_registry 
        09] [X] ailever_feature_store > remote_source_repository > remote_model_registry 
        10] [X] local_feature_store > ailever_source_repository > ailever_model_registry 
        11] [X] local_feature_store > ailever_source_repository > local_model_registry
        12] [X] local_feature_store > ailever_source_repository > remote_model_registry
        13] [X] local_feature_store > local_source_repository > ailever_model_registry 
        14] [X] local_feature_store > local_source_repository > local_model_registry 
        15] [X] local_feature_store > local_source_repository > remote_model_registry 
        16] [X] local_feature_store > remote_source_repository > ailever_model_registry 
        17] [X] local_feature_store > remote_source_repository > local_model_registry 
        18] [X] local_feature_store > remote_source_repository > remote_model_registry 
        19] [X] remote_feature_store > ailever_source_repository > ailever_model_registry 
        20] [X] remote_feature_store > ailever_source_repository > local_model_registry
        21] [X] remote_feature_store > ailever_source_repository > remote_model_registry
        22] [X] remote_feature_store > local_source_repository > ailever_model_registry 
        23] [X] remote_feature_store > local_source_repository > local_model_registry 
        24] [X] remote_feature_store > local_source_repository > remote_model_registry 
        25] [X] remote_feature_store > remote_source_repository > ailever_model_registry 
        26] [X] remote_feature_store > remote_source_repository > local_model_registry 
        27] [X] remote_feature_store > remote_source_repository > remote_model_registry 

        * prediction_trigger loading process : [02]
        01] [X] ailever_analysis_report_repository
        02] [O] local_analysis_report_repository
        03] [X] remote_analysis_report_repository

        """
        for security in baskets:
            # prediction_specification
            prediction_specifications[security]['ticker'] = security
            prediction_specification = prediction_specifications[security]
            framework = prediction_specification['framework']
            
            # initializing prediction_specification
            prediction_specification = self.trigger_block[framework].ui_buffer(prediction_specification, usage='prediction')
            
            # loading
            prediction_specification = self.fmr_manager.local_loading_connection(prediction_specification, usage='prediction')
            prediction_specification = self.fmr_manager.remote_loading_connection(prediction_specification, usage='prediction')
            self.trigger_block[framework].loaded_from(prediction_specification, usage='prediction')

            # prediction
            self.trigger_block[framework].predict(prediction_specification)
            
            # storing
            prediction_specification = self.fmr_manager.local_storing_connection(prediction_specification, usage='prediction')
            prediction_specification = self.fmr_manager.remote_storing_connection(prediction_specification, usage='prediction')
            self.trigger_block[framework].store_in(prediction_specification, usage='prediction')

    def analysis_trigger(self, baskets:list, analysis_specifications:dict):
        """
        * analysis_trigger loading process : []
        * analysis_trigger storing process : []
        """
        pass

    def evaluation_trigger(self, baskets:list, evaluation_specifications:dict):
        """
        * evaluation_trigger loading process : []
        * evaluation_trigger storing process : []
        """
        pass

    def available_models(self, baskets:list):
        for security in baskets:
            local_model_saving_informations = self.fmr_manager.local_finder(entity='ticker', target=security, framework=None) # list of dicts
            available_local_models = list(filter(lambda x: x[entity] == security, local_model_saving_informations))
            remote_model_saving_informations = self.fmr_manager.remote_finder(entity='ticker', target=security, framework=None) # list of dicts
            available_remote_models = list(filter(lambda x: x[entity] == security, remote_model_saving_informations))
            pprint(f'[AILEVER] Available {security} models in local system (L): ', available_local_models)
            pprint(f'[AILEVER] Available {security} models in remote system (R): ', available_remote_models)
            
    def forecasting_model_registry(self, command:str, framework:str=None):
        if command == 'listdir':
            return self._fmr_listdir(framework)
        elif command == 'listfiles':
            return self._fmr_listfiles(framework)
        elif command == 'remove':
            return self._fmr_remove(framework)
        elif command == 'clearall':
            return self._fmr_clearall(framework)

    def _fmr_listdir(self, framework:str=None):
        return self.fmr_manager.listdir(framework=framework)
    
    def _fmr_listfiles(self, framework:str=None):
        return self.fmr_manager.listfiles(framework=framework)

    def _fmr_remove(self, framework:str=None):
        pprint(self.fmr_manager.listfiles(framework=framework))
        id = int(input('ID : '))
        answer = input(f"Type 'Yes' if you really want to delete the model{id} in forecasting model registry.")
        if answer == 'Yes':
            model_saving_infomation = self.fmr_manager.local_finder(entity='id', target=id, framework=framework)
            self.fmr_manager.remove(name=model_saving_infomation['model_saving_name'], framework=framework)
    
    def _fmr_clearall(self, framework=None):
        answer = input(f"Type 'YES' if you really want to delete all models in forecasting model registry.")
        if answer == 'YES':
            self.fmr_manager.clearall(framework=framework)

    def analysis_report_repository(self, command:str, framework:str=None):
        pass

    def _arr_listdir(self, framework:str=None):
        pass

    def _arr_listfiles(self, framework:str=None):
        pass

    def _arr_remove(self, framework:str=None):
        pass
    
    def _arr_clearall(self, framework=None):
        pass

    def report(self, baskets:list):
        modelcore = self.trigger_block.ModelTransferCore()
        return modelcore

    def upload(self):
        pass

    def max_profit(self):
        pass

    def summary(self):
        pass



class Strategist:
    def __init__(self):
        pass

    def integrated_trigger(self):
        self.portfolio_trigger()
        self.backtesting_trigger()
    
    def portfolio_trigger(self):
        pass

    def backtesting_trigger(self):
        pass

    def strategy_model_registry(self, command:str, framework:str=None):
        pass

    def _smr_listdir(self, framework:str=None):
        pass

    def _smr_listfiles(self, framework:str=None):
        pass

    def _smr_remove(self, framework:str=None):
        pass

    def _smr_clearall(self, framework=None):
        pass

    def analysis_report_repository(self, command:str, framework:str=None):
        pass

    def _arr_listdir(self, framework:str=None):
        pass

    def _arr_listfiles(self, framework:str=None):
        pass

    def _arr_remove(self, framework:str=None):
        pass
    
    def _arr_clearall(self, framework=None):
        pass
