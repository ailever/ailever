from ailever.investment import __fmlops_bs__ as fmlops_bs
from .fmlops_management import FMR_Manager
from pprint import pprint

base_dir_core = dict()
base_dir_core['forecasting_model_registry'] = fmlops_bs.local_system.root.model_registry.forecasting_model_registry 

class Forecaster:
    def __init__(self, local_environment:dict=None, remote_environment:dict=None, framework:str='torch'):
        self.fmr_manager = FMR_Manager(core=base_dir_core['forecasting_model_registry'])

        if framework == 'torch':
            from ._base_trigger_blocks import TorchTriggerBlock
            self.trigger_block = TorchTriggerBlock(local_environment=local_environment, remote_environment=remote_environment)
        elif framework == 'tensorflow':
            from ._base_trigger_blocks import TensorflowTriggerBlock
            self.trigger_block = TensorflowForecaster(local_environment=local_environment, remote_environment=remote_environment)
        elif framework == 'sklearn':
            from ._base_trigger_blocks import SklearnTriggerBlock
            self.trigger_block = SklearnForecaster(local_environment=local_environment, remote_environment=remote_environment)
        elif framework == 'statsmodels':
            from ._base_trigger_blocks import StatsmodelsTriggerBlock
            self.trigger_block = StatsmodelsForecaster(local_environment=local_environment, remote_environment=remote_environment)
        else:
            assert False, '[AILEVER] The base framework for training models was not yet prepared.'

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
            train_specification = self.trigger_block.ui_buffer(train_specification)
            
            # loading
            train_specification = self.fmr_manager.loading_connection(train_specification)
            self.trigger_block.loaded_from(train_specification)

            # training
            self.trigger_block.train(train_specification)
            
            # storing
            train_specification = self.fmr_manager.storing_connection(train_specification)
            self.trigger_block.store_in(train_specification)
 
    def forecasting_model_registry(self):

    def remove(self, baskets:list, verbose:int=1):
        answer = input("Type 'Yes' if you really want to delete the baskets")
        if answer == 'Yes':
            for basket in baskets:
                core.remove(name=basket)
        else:
            return

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


