from .__base_structures import BaseForecaster
from ._base_transfer import ModelTransferCore
from .fmlops_nomenclatures import F_MRN

import torch

local_environment = dict()
local_environment['source_repository'] = 'source_repositry'
local_environment['model_registry'] = '.model_registry'
local_environment['model_loading_path'] = '.model_registry'  # priority 1
local_environment['model_saving_path'] = '.model_registry'   # priority 2

class TorchForecaster(BaseForecaster):
    def __init__(self, training_info:dict, local_environment:dict=local_environment, remote_environment:dict=None):
        self.local_environment = local_environment
        self.initializing_local_model_registry()

        self.remote_environment = remote_environment
        self.initializing_remote_model_registry()

        self.train()

        self.save_in_local_model_registry()
        self.save_in_remote_model_registry()

        self.prediction() 
        self.outcome_report()

    def initializing_local_model_registry(self):
        local_initialization_policy(self.local_environment)

    def initializing_remote_model_registry(self):
        remote_initialization_policy(self.remote_environment)
    
    def _setup_instances(self, source):
        from .fmlops_forecasters.torch._tensor_loader import train_dataloader, test_dataloader

        checkpoint = torch.load(os.path.join(source_repository['source_repository'], source))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if os.path.isdir(saving_directory):
            if os.path.isfile(os.path.join(saving_directory, saving_file)):
        checkpoint = torch.load('.models/' + training_info['saving_file'])
                training_info['first'] = checkpoint['first']
                training_info['cumulative_epochs'] = checkpoint['cumulative_epochs']

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

    def for_ModelTransferCore(self):
        pass



class TensorflowForecaster(BaseForecaster):
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

    def for_ModelTransferCore(self):
        pass



class SklearnForecaster(BaseForecaster):
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

    def for_ModelTransferCore(self):
        pass


class StatsmodelsForecaster(BaseForecaster):
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

    def for_ModelTransferCore(self):
        pass


def local_initialization_policy(local_environment):
    assert isinstance(local_environment, dict), 'The local_environment information must be supported by wtih dictionary data-type.'
    assert 'model_registry' in local_environment.keys(), 'Set your model repository path.'

    saving_directory = local_environment['model_registry']
    if not os.path.isdir(saving_directory):
        os.mkdir(saving_directory)

def remote_initialization_policy(remote_environment):
    pass

