from ailever.investment import __fmlops_bs__ as fmlops_bs
from .fmlops_management import *
from .trigger_blocks import TorchTriggerBlock, TensorflowTriggerBlock, SklearnTriggerBlock, StatsmodelsTriggerBlock

import re
from pprint import pprint

"""

* FMLOps Policy
- [FMLOPS] .fmlops
  |-- [FS] feature_store [Semi-Automation]
  |-- [SR] source_repository [Semi-Automation]
  |-- [MR] model_registry [Automation]
      |-- [FMR] forecasting_model_registry [Automation]
      |-- [SMR] strategy_model_registry [Heuristic Semi-Automation]
  |-- [ARR] analysis_report_repository [Heuristic Semi-Automation]
      |-- [FAR] fundamental_analysis_result
      |-- [TAR] technical_analysis_result
      |-- [MPR] model_prediction_result
      |-- [SAR] sector_analysis_result
  |-- [IOR] investment_outcome_repository [Automation]
      |-- [OPR] optimized_portfolio_registry
      |-- [BR] backtesting_repository
  |-- [MS] metadata_store [Automation]
      |-- [MS1] monitoring_source [Automation]
      |-- [DM] data_management
      |-- [MM] model_management
      |-- [MS2] model_specification


* Forecaster
- train_trigger loading process : [FS], [SR], [FMR]
- train_trigger storing process : [FMR], [MS3]
- prediction_trigger loading process : [FMR]
- prediction_trigger storing process : [MPR]
- analysis_trigger loading process : [FS]
- analysis_trigger storing process : [FAR], [TAR], [SAR]
- evaluation_trigger loading process : -
- evaluation_trigger storing process : -

* Strategist
- portfolio_trigger loading process : [FS], [FAR], [TAR], [MPR], [SAR]
- portfolio_trigger storing process : [OPR]
- backtesting_trigger loading process : [FS], [OPR]
- backtesting_trigger storing process : [BR]


"""

class Forecaster:
    def __init__(self, local_environment:dict=None, remote_environment:dict=None):
        self.__fs_manager = FS_Manager() # feature_store
        self.__sr_manager = SR_Manager() # source_repository
        self.__mr_manager = MR_Manager() # model_registry
        self.__fmr_manager = FMR_Manager() # forecasting_model_registry
        self.__far_manager = FAR_Manager() # fundamental_analysis_result
        self.__tar_manager = TAR_Manager() # technical_analysis_result
        self.__mpr_manager = MPR_Manager() # model_prediction_result
        self.__sar_manager = SAR_Manager() # sectore_analysis_result
        
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
        trigger = 'train'
        for security in baskets:
            # initializing train_specification
            train_specifications[security]['ticker'] = security
            train_specification = train_specifications[security]
            train_specification['__loading_process_regulation__'] = list()
            train_specification['__storing_process_regulation__'] = list()
            framework = train_specification['framework']
            train_specification = self.trigger_block[framework].ui_buffer(train_specification, usage=trigger)
            
            # loading
            train_specification = self.__fs_manager.loading_connection(train_specification, usage=trigger)
            train_specification = self.__sr_manager.loading_connection(train_specification, usage=trigger)
            train_specification = self.__fmr_manager.loading_connection(train_specification, usage=trigger)

            # trigger core : training
            self.trigger_block[framework].loaded_from(train_specification, usage=trigger)
            self.trigger_block[framework].train(train_specification)
            
            # storing
            train_specification = self.__fmr_manager.storing_connection(train_specification, usage=trigger)
            self.trigger_block[framework].store_in(train_specification, usage=trigger)
    
    def prediction_trigger(self, baskets:list, prediction_specifications:dict):
        trigger = 'prediction'
        for security in baskets:
            # initializing prediction_specification
            prediction_specifications[security]['ticker'] = security
            prediction_specification = prediction_specifications[security]
            prediction_specification['__loading_process_regulation__'] = list()
            prediction_specification['__storing_process_regulation__'] = list()
            framework = prediction_specification['framework']
            prediction_specification = self.trigger_block[framework].ui_buffer(prediction_specification, usage=trigger)
            
            # loading
            prediction_specification = self.__fmr_manager.loading_connection(prediction_specification, usage=trigger)

            # trigger core : prediction
            self.trigger_block[framework].loaded_from(prediction_specification, usage=trigger)
            self.trigger_block[framework].predict(prediction_specification)
            
            # storing
            prediction_specification = self.__fmr_manager.storing_connection(prediction_specification, usage=trigger)
            self.trigger_block[framework].store_in(prediction_specification, usage=trigger)

    def analysis_trigger(self, baskets:list, analysis_specifications:dict):
        pass

    def evaluation_trigger(self, baskets:list, evaluation_specifications:dict):
        pass

    def available_models(self, baskets:list):
        for security in baskets:
            local_model_saving_informations = self.__fmr_manager.local_finder(entity='ticker', target=security, framework=None) # list of dicts
            available_local_models = list(filter(lambda x: x[entity] == security, local_model_saving_informations))
            remote_model_saving_informations = self.__fmr_manager.remote_finder(entity='ticker', target=security, framework=None) # list of dicts
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
        return self.__fmr_manager.listdir(framework=framework)
    
    def _fmr_listfiles(self, framework:str=None):
        return self.__fmr_manager.listfiles(framework=framework)

    def _fmr_remove(self, framework:str=None):
        pprint(self.__fmr_manager.listfiles(framework=framework))
        id = int(input('ID : '))
        answer = input(f"Type 'Yes' if you really want to delete the model{id} in forecasting model registry.")
        if answer == 'Yes':
            model_saving_infomation = self.__fmr_manager.local_finder(entity='id', target=id, framework=framework)
            self.__fmr_manager.remove(name=model_saving_infomation['model_saving_name'], framework=framework)
    
    def _fmr_clearall(self, framework=None):
        answer = input(f"Type 'YES' if you really want to delete all models in forecasting model registry.")
        if answer == 'YES':
            self.__fmr_manager.clearall(framework=framework)

    def model_prediction_result(self, command:str, framework:str=None):
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
        self.__ior_manager = IOR_Manager() # investment_outcome_repository
        self.__opr_manager = OPR_Manager() # optimized_portfolio_registry

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
