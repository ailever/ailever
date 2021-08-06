from ._fmlops_policy import fmlops_bs

class Forecaster:
    def __init__(self, local_environment:dict=None, remote_environment:dict=None, framework:str='torch'):
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
        for security in baskets:
            self.trigger_block.train(train_specifications[security])
            self.trigger_block.save()

    def remove(self, baskets:list):
        answer = input("Type 'Yes' if you really want to delete the baskets")
        if answer == 'Yes':
            return
        else
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


