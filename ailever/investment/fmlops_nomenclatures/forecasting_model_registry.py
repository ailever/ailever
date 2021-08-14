from ailever.investment import __fmlops_bs__ as fmlops_bs
from ..__base_structures import BaseNomenclature

import datetime
from pytz import timezone
import re


class ForecastingModelRegistryNomenclature(BaseNomenclature):
    def __init__(self, core):
        self.core = core # fmlops_bs.local_system.root.model_registry.forecasting_model_registry 

        r"""
        model[id]_[framework]_[architecture]_[ticker]_[training_data_period_start]_[training_data_period_end]_[packet_size]_[perdiction_range]_v[version]_[rep]_[message]_[time]
        > model1_torch_lstm_are_20210324_20210324_365_100_v1_ailever_TargetingMarketCapital_2021_08_10
        """
        
        # saving policy
        self.id = 0
        self.framework = 'torch'
        self.architecture = 'lstm'
        self.ticker = 'are'
        self.training_data_period_start = '20200101'
        self.training_data_period_end = '20210801'
        self.packet_size = 365
        self.prediction_interval = 100
        self.version = 0
        self.rep = 'ailever'
        self.message = 'TargetingMarketCaptial'

    def __iter__(self):
        return self

    def __next__(self):
        if self.country == 'united_states':
            today = datetime.datetime.now(timezone('US/Eastern'))
        if self.country == 'korea':
            today = datetime.datetime.now(timezone('Asia/Seoul'))

        self.id += 1
        self.time = today.strftime(format='%Y%m%d-%H%M%S')

        id = self.id # [1]
        framework = self.framework # [2]
        architecture = self.architecture # [3]
        ticker = self.ticker # [4]
        training_data_period_start = self.training_data_period_start # [5]
        training_data_period_end = self.training_data_period_end # [6]
        packet_size = self.packet_size # [7]
        prediction_interval = self.prediction_interval # [8]
        version = self.version # [9]
        rep = self.rep # [10]
        message = self.message # [11]
        time = self.time # [12]

        name = f'model{id}_{framework}_{architecture}_{ticker}_{training_data_period_start}_{training_data_period_end}_{packet_size}_{prediction_interval}_v{version}_{rep}_{message}_{time}'
        return name

    def _management(self, framework):
        if framework == 'torch':
            model_saving_names = self.core.listdir(format='pt')
        
        self.model = dict()
        for model_saving_name in model_saving_names:
            pattern = '(.+)_'*12
            re_obj = re.search(pattern[:-1], model_saving_name)
            
            # saving information
            id = int(re_obj.group(1)[5:])
            self.model[id] = {'model_saving_name': model_saving_name,
                              'framework': re_obj.group(2),
                              'architecture': re_obj.group(3),
                              'ticker': re_obj.group(4),
                              'start': re_obj.group(5),
                              'end': re_obj.group(6),
                              'packet_size': re_obj.group(7),
                              'prediction_interval': re_obj.group(8),
                              'version': re_obj.group(9)[1:],
                              'rep': re_obj.group(10),
                              'message': re_obj.group(11),
                              'time': re_obj.group(12),
                              }
            

    def _search(self, entity, framework):
        self._management(framework=framework)
        if not self.model:
            if entity=='id':
                return 0
            elif entity=='version':
                return 1
        else:
            latest_id = max(self.model.keys())
            latest_version = self.model[latest_id]['version']
            if entity=='id':
                return latest_id
            elif entity=='version':
                return latest_version

    
    def loading_connection(self, train_specification):
        self._management(framework=train_specification['framework'])
        
        if not self.model.keys():
            train_specification['loading_model_name_from_local_model_registry'] = None
            return train_specification
        else:
            # ID exsistance in train_specification
            if 'id' in train_specification.keys():
                id = int(train_specification['id'])
                if id in self.model.keys():
                    train_specification['loading_model_name_from_local_model_registry'] = self.model[id]['model_saving_name']
                    return train_specification
            else:
                train_specification['loading_model_name_from_local_model_registry'] = None
                return train_specification

    def storing_connection(self, train_specification):
        self.country = train_specification['country']
        self.id = self._search(entity='id', framework=train_specification['framework']) # [1] : model1
        self.framework = train_specification['framework']                                # [2] : torch
        self.architecture = train_specification['architecture']                          # [3] : lstm00
        self.ticker = train_specification['ticker']                                      # [4] : ARE
        self.training_data_period_start = train_specification['start']                   # [5] : '20200101'
        self.training_data_period_end = train_specification['end']                       # [6] : '20210801'
        self.packet_size = train_specification['packet_size']                            # [7] : 365
        self.prediction_interval = train_specification['prediction_interval']            # [8] : 100
        self.version = self._search(entity='version', framework=train_specification['framework'])                                   # [9] : 1
        self.rep = train_specification['rep']                                            # [10] : ailever
        self.message = train_specification['message']                                    # [11] 'TargetingMarketCaptial'

        train_specification['saving_name_in_local_model_registry'] = next(self)
        return train_specification
