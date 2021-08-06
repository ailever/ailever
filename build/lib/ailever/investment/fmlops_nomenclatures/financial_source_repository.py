from ..__base_structures import BaseNomenclature

class SourceRepositoryNomenclature(BaseNomenclature):
    def __init__(self):
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
        self.version += 1

        architecture = self.architecture
        ticker = self.ticker
        training_data_period_start = self.training_data_period_start
        training_data_period_end = self.training_data_period_end
        packet_size = self.packet_size
        prediction_range = self.prediction_range
        version = self.version
        rep = self.rep
        message = self.message

        name = f'{architecture}_{ticker}_{training_data_period_start}_{training_data_period_end}_{packet_size}_{prediction_range}_v{version}_{rep}_{message}'
        return name
