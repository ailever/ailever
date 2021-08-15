from ailever.investment import __fmlops_bs__ as fmlops_bs
from .__base_structures import BaseTriggerBlock
from ._fmlops_policy import local_initialization_policy, remote_initialization_policy
from ._base_trigger_bridges import *
from ._base_transfer import ModelTransferCore

import sys
import os
from functools import partial

from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import torch


class TorchTriggerBlock(BaseTriggerBlock, TorchTriggerBridge):
    def __init__(self, local_environment:dict=None, remote_environment:dict=None):
        self.forecasting_model_registry = dict()
        self.strategy_model_registry = dict()
        self.analysis_report_repository = dict()
        sys.path.append(os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fmlops_forecasters'), 'torch'))

        if local_environment:
            self.local_environment = local_environment
            local_initialization_policy(self.local_environment)

        if remote_environment:
            self.remote_environment = remote_environment
            remote_initialization_policy(self.remote_environment)

    def ui_buffer(self, specification:dict, usage='train'):
        architecture = specification['architecture']
        UI_Transformation = self._dynamic_import(architecture, 'UI_Transformation')
        specification = UI_Transformation(specification)
        return specification

    def train(self, train_specification:dict):
        epochs = train_specification['epochs']
        device = train_specification['device']
        train_dataloader, test_dataloader, model, criterion, optimizer = self.instance_basis(train_specification, usage='train')

        for epoch in range(epochs):
            training_losses = []
            model.train()
            for batch_idx, (x_train, y_train) in enumerate(train_dataloader):
                # Forward
                hypothesis = model(x_train.squeeze().float().to(device))
                cost = criterion(hypothesis.squeeze().float().to(device), y_train.squeeze().float().to(device))
                # Backward
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                training_losses.append(cost)
            # Alert
            TrainMSE = torch.mean(torch.tensor(training_losses)).data
            if 'cumulative_epochs' in train_specification.keys():
                previous_cumulative_epochs = train_specification['cumulative_epochs']
                cumulative_epochs = train_specification['cumulative_epochs'] + epochs
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
                    hypothesis = model(x_train.squeeze().float().to(device))
                    cost = criterion(hypothesis.squeeze().float().to(device), y_train.squeeze().float().to(device))
                    validation_losses.append(cost)
                # Alert
                ValidationMSE = torch.mean(torch.tensor(validation_losses)).data
                if 'cumulative_epochs' in train_specification.keys():
                    previous_cumulative_epochs = train_specification['cumulative_epochs']
                    cumulative_epochs = train_specification['cumulative_epochs'] + epochs
                    print(f'[Validation][{epoch+1+previous_cumulative_epochs}/{cumulative_epochs}]', float(ValidationMSE))
                else:
                    print(f'[Validation][{epoch+1}/{epochs}]', float(ValidationMSE))
        
        # for saving
        self.forecasting_model_registry['model'] = model
        self.forecasting_model_registry['optimizer'] = optimizer
        self.forecasting_model_registry['epochs'] = train_specification['epochs']
        self.forecasting_model_registry['cumulative_epochs'] = train_specification['cumulative_epochs']+epochs if 'cumulative_epochs' in train_specification.keys() else epochs
        self.forecasting_model_registry['train_mse'] = TrainMSE
        self.forecasting_model_registry['validation_mse'] = ValidationMSE

    def predict(self, prediction_specification):
        scaler, investment_dataset, model = self.instance_basis(prediction_specification, usage='prediction')

        frame_last_packet = investment_dataset.frame_last_packet
        last_packet_data = investment_dataset.tensor_last_packet.to('cpu')
        packet_size = investment_dataset.packet_size
        train_range = investment_dataset.train_range
        prediction_range = investment_dataset.predict_range

        def predictor(spliter):
            raw_data = last_packet_data[spliter:spliter+train_range]
            normalized_raw_data, (mean, std) = scaler.standard(raw_data, return_statistics=True)
            result = model(normalized_raw_data.view(-1, *raw_data.size()).float()).squeeze()
            result = torch.cat([normalized_raw_data, result.detach()], dim=0)
            result = result*std + mean
            #result[-prediction_range:] = result[-prediction_range:] * 1.05
            return result.numpy()

        validation_packet = predictor(0)
        prediction_packet = predictor(packet_size-train_range)

        i = 0
        while True:
            i += 1
            date = pd.date_range(frame_last_packet.index[-train_range], periods=packet_size+i).to_frame()
            weekdays = date[(date.index.weekday != 5)&(date.index.weekday != 6)]
            if len(weekdays) == packet_size:
                prediction_packet_index = weekdays.index
                break

        # Result Table
        self.analysis_report_repository['prediction_table'] = pd.DataFrame(data=prediction_packet, columns=frame_last_packet.columns, index=prediction_packet_index).to_csv(os.path.join(saving_path, ticker) + '.csv')

        # Visualization
        plt.style.use('seaborn-whitegrid')
        def ploter(idx, column):
            _, axes = plt.subplots(2,1, figsize=(12,7))
            axes[0].plot(frame_last_packet.index, last_packet_data.numpy()[:,idx], lw=0, marker='o', c='black')
            axes[0].plot(frame_last_packet.index, validation_packet[:,idx])
            axes[0].axvline(frame_last_packet.index[packet_size-train_range], ls=':', c='r')
            axes[0].axvline(frame_last_packet.index[packet_size-1], ls=':', c='r')
            axes[0].set_title(ticker + f' {column} : Validation')
            axes[0].grid(True)

            axes[1].plot(prediction_packet_index[:train_range], last_packet_data.numpy()[:, idx][-train_range:], lw=0, marker='o', c='black')
            axes[1].plot(prediction_packet_index, prediction_packet[:, idx])
            axes[1].axvline(prediction_packet_index[0], ls=':', c='r')
            axes[1].axvline(prediction_packet_index[train_range-1], ls=':', c='r')
            axes[1].axvline(prediction_packet_index[-1], c='r')
            axes[1].set_title(ticker + f' {column} : Prediction')
            axes[1].grid(True)

            plt.legend()
            return deepcopy(plt)

        self.analysis_report_repository['prediction_visualization'] = dict()
        for idx, column in enumerate(frame_last_packet.columns):
            self.analysis_report_repository['prediction_visualization'][column] = ploter(idx, column)

    def analyze(self, analysis_specification):
        pass

    def evaluate(self, evalidation_specification):
        pass

    def loaded_from(self, specification:dict, usage='train'):
        trigger_loading_process = specification['loading_process']
        self = _loaded_from(self, trigger_loading_process, specification)

    def store_in(self, specification:dict, usage='train'):
        trigger_storing_process = specification['storing_process']
        self = _store_in(self, trigger_storing_process, specification)

    def outcome_report(self):
        pass

    def ModelTransferCore(self):
        modelcore = ModelTransferCore()
        return modelcore



class TensorflowTriggerBlock(BaseTriggerBlock, TensorflowTriggerBridge):
    def __init__(self, training_info:dict, local_environment:dict=None, remote_environment:dict=None):
        sys.path.append(os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fmlops_forecasters'), 'tensorflow'))
        if local_environment:
            local_initialization_policy(local_environment)
            self.local_environment = local_environmnet
            self.remote_environment = remote_environment

    def ui_buffer(self, specification:dict, usage='train'):
        return specification

    def train(self):
        pass

    def loaded_from(self, specification:dict, usage='train'):
        trigger_loading_process = specification['loading_process']
        self = _loaded_from(self, trigger_loading_process, usage=usage)       

    def store_in(self, specification:dict, usage='train'):
        trigger_storing_process = specification['storing_process']
        self = _store_in(self, trigger_storing_process, specification, usage=usage)       

    def predict(self):
        pass

    def outcome_report(self):
        pass

    def ModelTransferCore(self):
        modelcore = ModelTransferCore()
        return modelcore



class SklearnTriggerBlock(BaseTriggerBlock, SklearnTriggerBridge):
    def __init__(self, training_info:dict, local_environment:dict=None, remote_environment:dict=None):
        sys.path.append(os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fmlops_forecasters'), 'sklearn'))
        if local_environment:
            local_initialization_policy(local_environment)
            self.local_environment = local_environmnet
            self.remote_environment = remote_environment

    def ui_buffer(self, specification:dict, usage='train'):
        return specification

    def train(self):
        pass

    def loaded_from(self, specification:dict, usage='train'):
        trigger_loading_process = specification['loading_process']
        self = _loaded_from(self, trigger_loading_process, specification, usage=usage)       

    def store_in(self, specification:dict, usage='train'):
        trigger_storing_process = specification['storing_process']
        self = _store_in(self, trigger_storing_process, specification, usage=usage)       

    def predict(self):
        pass

    def outcome_report(self):
        pass

    def ModelTransferCore(self):
        modelcore = ModelTransferCore()
        return modelcore



class StatsmodelsTriggerBlock(BaseTriggerBlock, StatsmodelsTriggerBridge):
    def __init__(self, training_info:dict, local_environment:dict=None, remote_environment:dict=None):
        sys.path.append(os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fmlops_forecasters'), 'statsmodels'))
        if local_environment:
            local_initialization_policy(local_environment)
            self.local_environment = local_environmnet
            self.remote_environment = remote_environment

    def ui_buffer(self, specification:dict, usage='train'):
        return specification

    def train(self):
        pass

    def loaded_from(self, specification:dict, usage='train'):
        trigger_loading_process = specification['loading_process']
        self = _loaded_from(self, trigger_loading_process, specification, usage=usage)       

    def store_in(self, specification:dict, usage='train'):
        trigger_storing_process = specification['storing_process']
        self = _store_in(self, trigger_storing_process, specification, usage=usage)       

    def predict(self):
        pass

    def outcome_report(self):
        pass

    def ModelTransferCore(self):
        modelcore = ModelTransferCore()
        return modelcore


# loading path
def _loaded_from(self, trigger_loading_process:int, specification:dict, usage='train'):
    if usage == 'train':
        if trigger_loading_process == 1:
            self.load_from_ailever_feature_store(specification, usage)      # [1] feature_store
            self.load_from_ailever_source_repository(specification, usage)  # [2] source_repository
            self.load_from_ailever_model_registry(specification, usage)     # [3] model_registry
        elif trigger_loading_process == 2:
            self.load_from_ailever_feature_store(specification, usage)      # [1] feature_store
            self.load_from_ailever_source_repository(specification, usage)  # [2] source_repository
            self.load_from_local_model_registry(specification, usage)       # [3] model_registry
        elif trigger_loading_process == 3:
            self.load_from_ailever_feature_store(specification, usage)      # [1] feature_store
            self.load_from_ailever_source_repository(specification, usage)  # [2] source_repository
            self.load_from_remote_model_registry(specification, usage)      # [3] model_registry
        elif trigger_loading_process == 4:
            self.load_from_ailever_feature_store(specification, usage)      # [1] feature_store
            self.load_from_local_source_repository(specification, usage)    # [2] source_repository
            self.load_from_ailever_model_registry(specification, usage)     # [3] model_registry
        elif trigger_loading_process == 5:
            self.load_from_ailever_feature_store(specification, usage)      # [1] feature_store
            self.load_from_local_source_repository(specification, usage)    # [2] source_repository
            self.load_from_local_model_registry(specification, usage)       # [3] model_registry
        elif trigger_loading_process == 6:
            self.load_from_ailever_feature_store(specification, usage)      # [1] feature_store
            self.load_from_local_source_repository(specification, usage)    # [2] source_repository
            self.load_from_remote_model_registry(specification, usage)      # [3] model_registry
        elif trigger_loading_process == 7:
            self.load_from_ailever_feature_store(specification, usage)      # [1] feature_store
            self.load_from_remote_source_repository(specification, usage)   # [2] source_repository
            self.load_from_ailever_model_registry(specification, usage)     # [3] model_registry
        elif trigger_loading_process == 8:
            self.load_from_ailever_feature_store(specification, usage)      # [1] feature_store
            self.load_from_remote_source_repository(specification, usage)   # [2] source_repository
            self.load_from_local_model_registry(specification, usage)       # [3] model_registry
        elif trigger_loading_process == 9:
            self.load_from_ailever_feature_store(specification, usage)      # [1] feature_store
            self.load_from_remote_source_repository(specification, usage)   # [2] source_repository
            self.load_from_remote_model_registry(specification, usage)      # [3] model_registry
        elif trigger_loading_process == 10:
            self.load_from_local_feature_store(specification, usage)        # [1] feature_store
            self.load_from_ailever_source_repository(specification, usage)  # [2] source_repository
            self.load_from_ailever_model_registry(specification, usage)     # [3] model_registry
        elif trigger_loading_process == 11:
            self.load_from_local_feature_store(specification, usage)        # [1] feature_store
            self.load_from_ailever_source_repository(specification, usage)  # [2] source_repository
            self.load_from_local_model_registry(specification, usage)       # [3] model_registry
        elif trigger_loading_process == 12:
            self.load_from_local_feature_store(specification, usage)        # [1] feature_store
            self.load_from_ailever_source_repository(specification, usage)  # [2] source_repository
            self.load_from_remote_model_registry(specification, usage)      # [3] model_registry
        elif trigger_loading_process == 13:
            self.load_from_local_feature_store(specification, usage)        # [1] feature_store
            self.load_from_local_source_repository(specification, usage)    # [2] source_repository
            self.load_from_ailever_model_registry(specification, usage)     # [3] model_registry
        elif trigger_loading_process == 14:
            self.load_from_local_feature_store(specification, usage)        # [1] feature_store
            self.load_from_local_source_repository(specification, usage)    # [2] source_repository
            self.load_from_local_model_registry(specification, usage)       # [3] model_registry
        elif trigger_loading_process == 15:
            self.load_from_local_feature_store(specification, usage)        # [1] feature_store
            self.load_from_local_source_repository(specification, usage)    # [2] source_repository
            self.load_from_remote_model_registry(specification, usage)      # [3] model_registry
        elif trigger_loading_process == 16:
            self.load_from_local_feature_store(specification, usage)        # [1] feature_store
            self.load_from_remote_source_repository(specification, usage)   # [2] source_repository
            self.load_from_ailever_model_registry(specification, usage)     # [3] model_registry
        elif trigger_loading_process == 17:
            self.load_from_local_feature_store(specification, usage)        # [1] feature_store
            self.load_from_remote_source_repository(specification, usage)   # [2] source_repository
            self.load_from_local_model_registry(specification, usage)       # [3] model_registry
        elif trigger_loading_process == 18:
            self.load_from_local_feature_store(specification, usage)        # [1] feature_store
            self.load_from_remote_source_repository(specification, usage)   # [2] source_repository
            self.load_from_remote_model_registry(specification, usage)      # [3] model_registry
        elif trigger_loading_process == 19:
            self.load_from_remote_feature_store(specification, usage)       # [1] feature_store
            self.load_from_ailever_source_repository(specification, usage)  # [2] source_repository
            self.load_from_ailever_model_registry(specification, usage)     # [3] model_registry
        elif trigger_loading_process == 20:
            self.load_from_remote_feature_store(specification, usage)       # [1] feature_store
            self.load_from_ailever_source_repository(specification, usage)  # [2] source_repository
            self.load_from_local_model_registry(specification, usage)       # [3] model_registry
        elif trigger_loading_process == 21:
            self.load_from_remote_feature_store(specification, usage)       # [1] feature_store
            self.load_from_ailever_source_repository(specification, usage)  # [2] source_repository
            self.load_from_remote_model_registry(specification, usage)      # [3] model_registry
        elif trigger_loading_process == 22:
            self.load_from_remote_feature_store(specification, usage)       # [1] feature_store
            self.load_from_local_source_repository(specification, usage)    # [2] source_repository
            self.load_from_ailever_model_registry(specification, usage)     # [3] model_registry
        elif trigger_loading_process == 23:
            self.load_from_remote_feature_store(specification, usage)       # [1] feature_store
            self.load_from_local_source_repository(specification, usage)    # [2] source_repository
            self.load_from_local_model_registry(specification, usage)       # [3] model_registry
        elif trigger_loading_process == 24:
            self.load_from_remote_feature_store(specification, usage)       # [1] feature_store
            self.load_from_local_source_repository(specification, usage)    # [2] source_repository
            self.load_from_remote_model_registry(specification, usage)      # [3] model_registry
        elif trigger_loading_process == 25:
            self.load_from_remote_feature_store(specification, usage)       # [1] feature_store
            self.load_from_remote_source_repository(specification, usage)   # [2] source_repository
            self.load_from_ailever_model_registry(specification, usage)     # [3] model_registry
        elif trigger_loading_process == 26:
            self.load_from_remote_feature_store(specification, usage)       # [1] feature_store
            self.load_from_remote_source_repository(specification, usage)   # [2] source_repository
            self.load_from_local_model_registry(specification, usage)       # [3] model_registry
        elif trigger_loading_process == 27:
            self.load_from_remote_feature_store(specification, usage)       # [1] feature_store
            self.load_from_remote_source_repository(specification, usage)   # [2] source_repository
            self.load_from_remote_model_registry(specification, usage)      # [3] model_registry
    return self

# storing_path
def _store_in(self, trigger_storing_process:int, specification:dict, usage:str='train'):
    if usage == 'train':
        if trigger_storing_process == 1:
            self.save_in_ailever_feature_store(specification, usage)   # [1] feature_store
            self.save_in_ailever_model_registry(specification, usage)  # [2] model_registry
            self.save_in_ailever_metadata_store(specification, usage)  # [3] metadata_store
        elif trigger_storing_process == 2:
            self.save_in_ailever_feature_store(specification, usage)   # [1] feature_store
            self.save_in_ailever_model_registry(specification, usage)  # [2] model_registry
            self.save_in_local_metadata_store(specification, usage)    # [3] metadata_store
        elif trigger_storing_process == 3:
            self.save_in_ailever_feature_store(specification, usage)   # [1] feature_store
            self.save_in_ailever_model_registry(specification, usage)  # [2] model_registry
            self.save_in_remote_metadata_store(specification, usage)   # [3] metadata_store
        elif trigger_storing_process == 4:
            self.save_in_ailever_feature_store(specification, usage)   # [1] feature_store
            self.save_in_local_model_registry(specification, usage)    # [2] model_registry
            self.save_in_ailever_metadata_store(specification, usage)  # [3] metadata_store
        elif trigger_storing_process == 5:
            self.save_in_ailever_feature_store(specification, usage)   # [1] feature_store
            self.save_in_local_model_registry(specification, usage)    # [2] model_registry
            self.save_in_local_metadata_store(specification, usage)    # [3] metadata_store
        elif trigger_storing_process == 6:
            self.save_in_ailever_feature_store(specification, usage)   # [1] feature_store
            self.save_in_local_model_registry(specification, usage)    # [2] model_registry
            self.save_in_remote_metadata_store(specification, usage)   # [3] metadata_store
        elif trigger_storing_process == 7:
            self.save_in_ailever_feature_store(specification, usage)   # [1] feature_store
            self.save_in_remote_model_registry(specification, usage)   # [2] model_registry
            self.save_in_ailever_metadata_store(specification, usage)  # [3] metadata_store
        elif trigger_storing_process == 8:
            self.save_in_ailever_feature_store(specification, usage)   # [1] feature_store
            self.save_in_remote_model_registry(specification, usage)   # [2] model_registry
            self.save_in_local_metadata_store(specification, usage)    # [3] metadata_store
        elif trigger_storing_process == 9:
            self.save_in_ailever_feature_store(specification, usage)   # [1] feature_store
            self.save_in_remote_model_registry(specification, usage)   # [2] model_registry
            self.save_in_remote_metadata_store(specification, usage)   # [3] metadata_store
        elif trigger_storing_process == 10:
            self.save_in_local_feature_store(specification, usage)     # [1] feature_store
            self.save_in_ailever_model_registry(specification, usage)  # [2] model_registry
            self.save_in_ailever_metadata_store(specification, usage)  # [3] metadata_store
        elif trigger_storing_process == 11:
            self.save_in_local_feature_store(specification, usage)     # [1] feature_store
            self.save_in_ailever_model_registry(specification, usage)  # [2] model_registry
            self.save_in_local_metadata_store(specification, usage)    # [3] metadata_store
        elif trigger_storing_process == 12:
            self.save_in_local_feature_store(specification, usage)     # [1] feature_store
            self.save_in_ailever_model_registry(specification, usage)  # [2] model_registry
            self.save_in_remote_metadata_store(specification, usage)   # [3] metadata_store
        elif trigger_storing_process == 13:
            self.save_in_local_feature_store(specification, usage)     # [1] feature_store
            self.save_in_local_model_registry(specification, usage)    # [2] model_registry
            self.save_in_ailever_metadata_store(specification, usage)  # [3] metadata_store
        elif trigger_storing_process == 14:
            self.save_in_local_feature_store(specification, usage)     # [1] feature_store
            self.save_in_local_model_registry(specification, usage)    # [2] model_registry
            self.save_in_local_metadata_store(specification, usage)    # [3] metadata_store
        elif trigger_storing_process == 15:
            self.save_in_local_feature_store(specification, usage)     # [1] feature_store
            self.save_in_local_model_registry(specification, usage)    # [2] model_registry
            self.save_in_remote_metadata_store(specification, usage)   # [3] metadata_store
        elif trigger_storing_process == 16:
            self.save_in_local_feature_store(specification, usage)     # [1] feature_store
            self.save_in_remote_model_registry(specification, usage)   # [2] model_registry
            self.save_in_ailever_metadata_store(specification, usage)  # [3] metadata_store
        elif trigger_storing_process == 17:
            self.save_in_local_feature_store(specification, usage)     # [1] feature_store
            self.save_in_remote_model_registry(specification, usage)   # [2] model_registry
            self.save_in_local_metadata_store(specification, usage)    # [3] metadata_store
        elif trigger_storing_process == 18:
            self.save_in_local_feature_store(specification, usage)     # [1] feature_store
            self.save_in_remote_model_registry(specification, usage)   # [2] model_registry
            self.save_in_remote_metadata_store(specification, usage)   # [3] metadata_store
        elif trigger_storing_process == 19:
            self.save_in_remote_feature_store(specification, usage)    # [1] feature_store
            self.save_in_ailever_model_registry(specification, usage)  # [2] model_registry
            self.save_in_ailever_metadata_store(specification, usage)  # [3] metadata_store
        elif trigger_storing_process == 20:
            self.save_in_remote_feature_store(specification, usage)    # [1] feature_store
            self.save_in_ailever_model_registry(specification, usage)  # [2] model_registry
            self.save_in_local_metadata_store(specification, usage)    # [3] metadata_store
        elif trigger_storing_process == 21:
            self.save_in_remote_feature_store(specification, usage)    # [1] feature_store
            self.save_in_ailever_model_registry(specification, usage)  # [2] model_registry
            self.save_in_remote_metadata_store(specification, usage)   # [3] metadata_store
        elif trigger_storing_process == 22:
            self.save_in_remote_feature_store(specification, usage)    # [1] feature_store
            self.save_in_local_model_registry(specification, usage)    # [2] model_registry
            self.save_in_ailever_metadata_store(specification, usage)  # [3] metadata_store
        elif trigger_storing_process == 23:
            self.save_in_remote_feature_store(specification, usage)    # [1] feature_store
            self.save_in_local_model_registry(specification, usage)    # [2] model_registry
            self.save_in_local_metadata_store(specification, usage)    # [3] metadata_store
        elif trigger_storing_process == 24:
            self.save_in_remote_feature_store(specification, usage)    # [1] feature_store
            self.save_in_local_model_registry(specification, usage)    # [2] model_registry
            self.save_in_remote_metadata_store(specification, usage)   # [3] metadata_store
        elif trigger_storing_process == 25:
            self.save_in_remote_feature_store(specification, usage)    # [1] feature_store
            self.save_in_remote_model_registry(specification, usage)   # [2] model_registry
            self.save_in_ailever_metadata_store(specification, usage)  # [3] metadata_store
        elif trigger_storing_process == 26:
            self.save_in_remote_feature_store(specification, usage)    # [1] feature_store
            self.save_in_remote_model_registry(specification, usage)   # [2] model_registry
            self.save_in_local_metadata_store(specification, usage)    # [3] metadata_store
        elif trigger_storing_process == 27:
            self.save_in_remote_feature_store(specification, usage)    # [1] feature_store
            self.save_in_remote_model_registry(specification, usage)   # [2] model_registry
            self.save_in_remote_metadata_store(specification, usage)   # [3] metadata_store
    return self
