from ..__base_structures import BaseNomenclature

class ModelRegistryNomenclature(BaseNomenclature):
    def __init__(self):
        r"""
        model[id]_[architecture]_[ticker]_[training_data_period_start]_[training_data_period_end]_[packet_size]_[perdiction_range]_v[version]_[rep]_[message]
        > model1_lstm_are_20210324_20210324_365_100_v1_ailever_TargetingMarketCapital
        """
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

    def __iter__(self):
        return self

    def __next__(self):
        self.id += 1
        self.version += 1

        id = self.id
        architecture = self.architecture
        ticker = self.ticker
        training_data_period_start = self.training_data_period_start
        training_data_period_end = self.training_data_period_end
        packet_size = self.packet_size
        prediction_range = self.prediction_range
        version = self.version
        rep = self.rep
        message = self.message

        name = f'model{id}_{architecture}_{ticker}_{training_data_period_start}_{training_data_period_end}_{packet_size}_{prediction_range}_v{version}_{rep}_{message}'
        return name
