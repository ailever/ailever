from ailever.investment import fmlops_bs
from ..__base_structures import BaseNomenclature

import datetime
from pytz import timezone
import re

class ModelRegistryNomenclature(BaseNomenclature):
    def __init__(self):
        r"""
        model[id]_[architecture]_[ticker]_[training_data_period_start]_[training_data_period_end]_[packet_size]_[perdiction_range]_v[version]_[rep]_[message]_[time]
        > model1_lstm_are_20210324_20210324_365_100_v1_ailever_TargetingMarketCapital_2021_08_10
        """

        self.id = 0
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

        id = self.id #1
        architecture = self.architecture #2
        ticker = self.ticker #3
        training_data_period_start = self.training_data_period_start #4
        training_data_period_end = self.training_data_period_end #5
        packet_size = self.packet_size #6
        prediction_interval = self.prediction_interval #7
        version = self.version #8
        rep = self.rep #9
        message = self.message #10
        time = self.time #11

        name = f'model{id}_{architecture}_{ticker}_{training_data_period_start}_{training_data_period_end}_{packet_size}_{prediction_interval}_v{version}_{rep}_{message}_{time}'
        return name
    
    def search(self, entity):
        ids = list()
        versions = list()

        models = fmlops_bs.local_system.root.model_registry.listdir(format='pt')
        if not models:
            if entity=='id':
                return 0
            elif entity=='version':
                return 1
        else:
            for model in models:
                pattern = '(.+)_'*11
                re_obj = re.search(pattern[:-1], model)
                ids.append(re_obj.group(1)[5:])
                versions.append(re_obj.group(8)[1:])
        
        if entity=='id':
            return max(ids)
        elif entity=='version':
            return max(versions)

    def connect(self, train_specification):
        self.country = train_specification['country']

        self.id = self.search(entity='id')
        self.architecture = train_specification['architecture'] # lstm00
        self.ticker = train_specification['ticker'] # ARE
        self.training_data_period_start = train_specification['start']# '20200101'
        self.training_data_period_end = train_specification['end']# '20210801'
        self.packet_size = train_specification['packet_size'] # 365
        self.prediction_interval = train_specification['prediction_interval'] # 100
        self.version = self.search(entity='version')
        self.rep = train_specification['rep'] # ailever
        self.message = train_specification['message'] # 'TargetingMarketCaptial'

        train_specification['saving_name'] = next(self)
        return train_specification
