from .__base_structures import BaseTriggerBlock
from ._base_transfer import ModelTransferCore
from .fmlops_nomenclatures import F_MRN

import torch
from importlib import import_module


class TorchTriggerBlock(BaseTriggerBlock):
    def __init__(self, training_info:dict, local_environment:dict=local_environment, remote_environment:dict=None):
        self.local_environment = local_environment
        self.initializing_local_model_registry()

        self.remote_environment = remote_environment
        self.initializing_remote_model_registry()

        self.prediction() 
        self.outcome_report()

    def initializing_local_model_registry(self):
        local_initialization_policy(self.local_environment)

    def initializing_remote_model_registry(self):
        remote_initialization_policy(self.remote_environment)
    
    def instance_basis(self, source):
        from .fmlops_forecasters.torch.lstm import InvestmentDataset, InvestmentDataLoader, Model, Criterion, Optimizer
        getattr(import_module('.fmlops_forecasters.torch.lstm'), '', '', '', '', '')
        dataset = InvestmentDataset()
        train_dataloader, test_dataloader = InvestmentDataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        model = Model()
        criterion = Criterion()
        optimizer = Optimizer()

        checkpoint = torch.load(os.path.join(source_repository['source_repository'], source))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return train_dataloader, test_dataloader, model, criterion, optimizer

    def train(self):
        train_dataloader, test_dataloader, model, criterion, optimizer = self._setup_instances()
        for epoch in range(epochs):
            training_losses = []
            model.train()
            for batch_idx, (x_train, y_train) in enumerate(train_dataloader):
                # Forward
                hypothesis = model(x_train.squeeze().float().to(training_info['device']))
                cost = criterion(hypothesis.squeeze().float().to(training_info['device']), y_train.squeeze().float().to(training_info['device']))
                # Backward
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                training_losses.append(cost)
            # Alert
            TrainMSE = torch.mean(torch.tensor(training_losses)).data
            if 'cumulative_epochs' in training_info.keys():
                previous_cumulative_epochs = training_info['cumulative_epochs']
                cumulative_epochs = training_info['cumulative_epochs'] + epochs
                print(f'[Training  ][{epoch+1+previous_cumulative_epochs}/{cumulative_epochs}]', float(TrainMSE))
            else:
                print(f'[Training  ][{epoch+1}/{epochs}]', float(TrainMSE))
            
            if torch.isnan(TrainMSE).all():
                break 

            with torch.no_grad():
                validation_losses = []
                model.eval()
                for batch_idx, (x_train, y_train) in enumerate(test_dataloader):
                    # Forward
                    hypothesis = model(x_train.squeeze().float().to(training_info['device']))
                    cost = criterion(hypothesis.squeeze().float().to(training_info['device']), y_train.squeeze().float().to(training_info['device']))
                    validation_losses.append(cost)
                # Alert
                ValidationMSE = torch.mean(torch.tensor(validation_losses)).data
                if 'cumulative_epochs' in training_info.keys():
                    previous_cumulative_epochs = training_info['cumulative_epochs']
                    cumulative_epochs = training_info['cumulative_epochs'] + epochs
                    print(f'[Validation][{epoch+1+previous_cumulative_epochs}/{cumulative_epochs}]', float(ValidationMSE))
                else:
                    print(f'[Validation][{epoch+1}/{epochs}]', float(ValidationMSE))

        self.save_in_local_model_registry()
        self.save_in_remote_model_registry()

    def prediction(self):
        pass

    def load_from_ailever_source_repository(self):
        return None

    def load_from_local_source_repository(self):
        return None

    def load_from_local_model_registry(self):
        return None

    def load_from_remote_source_repository(self):
        return None

    def load_from_remote_model_registry(self):
        return None

    def save_in_local_model_registry(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def ModelTransferCore(self):
        modelcore = ModelTransferCore()
        return modelcore



class TensorflowTriggerBlock(BaseTriggerBlock):
    def __init__(self, training_info:dict, local_environment:dict=local_environment, remote_environment:dict=None):
        self.local_environment = local_environmnet
        self.remote_environment = remote_environment

    def initializing_local_model_registry(self):
        local_initialization_policy(self.local_environment)

    def initializing_remote_model_registry(self):
        remote_initialization_policy(self.remote_environment)

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_ailever_source_repository(self):
        return None

    def load_from_local_source_repository(self):
        return None

    def load_from_local_model_registry(self):
        return None

    def load_from_remote_source_repository(self):
        return None

    def load_from_remote_model_registry(self):
        return None

    def save_in_local_model_registry(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def ModelTransferCore(self):
        modelcore = ModelTransferCore()
        return modelcore



class SklearnTriggerBlock(BaseTriggerBlock):
    def __init__(self, training_info:dict, local_environment:dict=local_environment, remote_environment:dict=None):
        self.local_environment = local_environmnet
        self.remote_environment = remote_environment

    def initializing_local_model_registry(self):
        local_initialization_policy(self.local_environment)

    def initializing_remote_model_registry(self):
        remote_initialization_policy(self.remote_environment)

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_ailever_source_repository(self):
        return None

    def load_from_local_source_repository(self):
        return None

    def load_from_local_model_registry(self):
        return None

    def load_from_remote_source_repository(self):
        return None

    def load_from_remote_model_registry(self):
        return None

    def save_in_local_model_registry(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def ModelTransferCore(self):
        modelcore = ModelTransferCore()
        return modelcore

class StatsmodelsTriggerBlock(BaseTriggerBlock):
    def __init__(self, training_info:dict, local_environment:dict=local_environment, remote_environment:dict=None):
        self.local_environment = local_environmnet
        self.remote_environment = remote_environment

    def initializing_local_model_registry(self):
        local_initialization_policy(self.local_environment)

    def initializing_remote_model_registry(self):
        remote_initialization_policy(self.remote_environment)

    def train(self):
        pass

    def prediction(self):
        pass

    def load_from_ailever_source_repository(self):
        return None

    def load_from_local_source_repository(self):
        return None

    def load_from_local_model_registry(self):
        return None

    def load_from_remote_source_repository(self):
        return None

    def load_from_remote_model_registry(self):
        return None

    def save_in_local_model_registry(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def ModelTransferCore(self):
        modelcore = ModelTransferCore()
        return modelcore


def local_initialization_policy(local_environment):
    assert isinstance(local_environment, dict), 'The local_environment information must be supported by wtih dictionary data-type.'
    assert 'model_registry' in local_environment.keys(), 'Set your model repository path.'

    saving_directory = local_environment['model_registry']
    if not os.path.isdir(saving_directory):
        os.mkdir(saving_directory)

def remote_initialization_policy(remote_environment):
    pass

