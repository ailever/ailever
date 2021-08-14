from ailever.investment import __fmlops_bs__ as fmlops_bs
from .fmlops_management import FMR_Manager
from ._base_trigger_blocks import TorchTriggerBlock, TensorflowTriggerBlock, SklearnTriggerBlock, StatsmodelsTriggerBlock
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

    def train_trigger(self, baskets:list, train_specifications:dict):
        """
        * trigger loading process : [02]
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

        * trigger storing process : [14]
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
            train_specification = self.trigger_block[framework].ui_buffer(train_specification)
            
            # loading
            train_specification = self.fmr_manager.loading_connection(train_specification)
            self.trigger_block[framework].loaded_from(train_specification)

            # training
            self.trigger_block[framework].train(train_specification)
            
            # storing
            train_specification = self.fmr_manager.storing_connection(train_specification)
            self.trigger_block[framework].store_in(train_specification)
 
    def model_registry(self, command:str, frmaework:str=None):
        if command == 'listdir':
            return self._listdir(framework)
        elif command == 'listfiles':
            return self._listfiles(framework)
        elif command == 'remove':
            return self._remove(framework)
        elif command == 'clearall':
            return self._clearall(framework)

    def _listdir(self, frmaework:str=None):
        return self.fmr_manager.listdir(framework=framework)
    
    def _listfiles(self, framework:str=None):
        return self.fmr_manager.listfiles(framework=framework)

    def _remove(self, framework:str=None):
        pprint(self.fmr_manager.listfiles(framework=framework))
        id = int(input('ID : '))
        answer = input(f"Type 'Yes' if you really want to delete the model{id} in forecasting model registry.")
        if answer == 'Yes':
            file_name = self.fmr_manager.finder(entity='id', target=id, framework=framework)
            self.fmr_manager.remove(name=file_name, framework=framework)
    
    def _clearall(self, framework=None):
        pprint(self.fmr_manager.listdir(framework=framework))
        answer = input(f"Type 'YES' if you really want to delete all models in forecasting model registry.")
        if answer == 'YES':
            self.fmr_manager.clearall(framework=framework)

    def evaluation_trigger(self):
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


