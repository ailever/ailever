from ..__base_structures import BaseNomenclature

import datetime
from pytz import timezone

class ModelRegistryNomenclature(BaseNomenclature):
    def __init__(self, country='korea'):
        r"""
        model[id]_[architecture]_[ticker]_[training_data_period_start]_[training_data_period_end]_[packet_size]_[perdiction_range]_v[version]_[rep]_[message]_[time]
        > model1_lstm_are_20210324_20210324_365_100_v1_ailever_TargetingMarketCapital_2021_08_10
        """
        if country == 'united states':
            today = datetime.datetime.now(timezone('US/Eastern'))
        if country == 'korea':
            today = datetime.datetime.now(timezone('Asia/Seoul'))

        self.id = 0
        self.architecture = 'lstm'
        self.ticker = 'are'
        self.training_data_period_start = '20200101'
        self.training_data_period_end = '20210801'
        self.packet_size = 365
        self.prediction_range = 100
        self.version = 0
        self.rep = 'ailever'
        self.message = 'TargetingMarketCaptial'
        self.time = today.strftime(format='%Y%m%d-%H%M%S')

    def __iter__(self):
        return self

    def __next__(self):
        self.id += 1
        self.version += 1

        id = self.id #1
        architecture = self.architecture #2
        ticker = self.ticker #3
        training_data_period_start = self.training_data_period_start #4
        training_data_period_end = self.training_data_period_end #5
        packet_size = self.packet_size #6
        prediction_range = self.prediction_range #7
        version = self.version #8
        rep = self.rep #9
        message = self.message #10
        time = self.time #11

        name = f'model{id}_{architecture}_{ticker}_{training_data_period_start}_{training_data_period_end}_{packet_size}_{prediction_range}_v{version}_{rep}_{message}_{time}'
        return name

