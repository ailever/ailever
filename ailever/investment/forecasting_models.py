local_environment = dict()
local_environment['source_repository'] = 'source_repositry'
local_environment['model_registry'] = '.model_registry'
local_environment['model_loading_path'] = '.model_registry'  # priority 1
local_environment['model_saving_path'] = '.model_registry'   # priority 2


class Forecaster:
    def __init__(self, local_environment:dict, remote_environment:dict=None, framework='torch'):
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

    def train_trigger(self, baskets:list):
        for security in baskets:
            self.trigger_block.train(security)

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



