from abc import *
from importlib import import_module
import os
import re
from datetime import datetime
from shutil import copyfile
from copy import deepcopy
import numpy as np
import pandas as pd
import joblib


class Framework(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_model_class(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def upload(self):
        pass
 
    @abstractmethod
    def save_insidemodel(self):
        pass

    @abstractmethod
    def save_outsidemodel(self):
        pass


class FrameworkSklearn(Framework):
    def __init__(self):
        self.modules = dict()
        self.modules['linear_model'] = list(filter(
            lambda x: re.search('Classifier|Regression|Regressor', x), import_module('sklearn.linear_model').__all__))
        self.modules['ensemble'] = list(filter(
            lambda x: re.search('Classifier|Regressor', x), import_module('sklearn.ensemble').__all__))
        self.modules['neighbors'] = list(filter(
            lambda x: re.search('Classifier|Regressor', x), import_module('sklearn.neighbors').__all__))
        self.modules['tree'] = list(filter(
            lambda x: re.search('Classifier|Regressor', x), import_module('sklearn.tree').__all__))
        self.modules['svm'] = list(filter(
            lambda x: re.search('SVC|SVR', x), import_module('sklearn.svm').__all__))
 
        self.models = list()
        for model_set in self.modules.values():
            self.models.extend(model_set)

    def get_model_class(self, supported_framework, module_name, model_name):
        model_class = getattr(import_module(supported_framework+'.'+module_name), model_name)
        return model_class

    def train(self, model, dataset):
        X = dataset.loc[:, dataset.columns != 'target']
        y = dataset.loc[:, 'target'].ravel()
        model.fit(X, y)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def upload(self, model_registry_path):
        return joblib.load(model_registry_path)

    def save_insidemodel(self, model, mlops_path, saving_name):
        saving_name = saving_name + '.joblib'
        model_registry_saving_path = os.path.join(mlops_path, saving_name)
        joblib.dump(model, model_registry_saving_path)

        training_info_detail = dict()
        training_info_detail['saving_model_name'] = saving_name
        return training_info_detail

    def save_outsidemodel(self, model, model_registry_path, outsidelog_path):
        extension = '.joblib'
        model_registry_path = model_registry_path + extension
        outsidelog = pd.read_csv(outsidelog_path)
        outsidelog.iat[0, 3] = outsidelog.iat[0, 3] + extension
        outsidelog.to_csv(outsidelog_path, index=False)
        return joblib.dump(model, model_registry_path)


class FrameworkXgboost(Framework):
    def __init__(self):
        self.modules = dict()
        self.modules['xgboost_model'] = list(filter(lambda x: re.search('Classifier|Regressor', x), import_module('xgboost').__all__))

        self.models = list()
        for model_set in self.modules.values():
            self.models.extend(model_set)

    def get_model_class(self, supported_framework, module_name, model_name):
        model_class = getattr(import_module(supported_framework), model_name)
        return model_class

    def train(self, model, dataset):
        X = dataset.loc[:, dataset.columns != 'target']
        y = dataset.loc[:, 'target'].ravel()
        model.fit(X, y)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def upload(self, model_registry_path):
        return joblib.load(model_registry_path)

    def save_insidemodel(self, model, mlops_path, saving_name):
        saving_name = saving_name + '.joblib'
        model_registry_saving_path = os.path.join(mlops_path, saving_name)
        joblib.dump(model, model_registry_saving_path)

        training_info_detail = dict()
        training_info_detail['saving_model_name'] = saving_name
        return training_info_detail

    def save_outsidemodel(self, model, model_registry_path, outsidelog_path):
        extension = '.joblib'
        model_registry_path = model_registry_path + extension
        outsidelog = pd.read_csv(outsidelog_path)
        outsidelog.iat[0, 3] = outsidelog.iat[0, 3] + extension
        outsidelog.to_csv(outsidelog_path, index=False)
        return joblib.dump(model, model_registry_path)

class FrameworkLightgbm(Framework):
    def __init__(self):
        self.modules = dict()
        self.modules['lightgbm_model'] = list(filter(lambda x: re.search('Classifier|Regressor', x), import_module('lightgbm').__all__))

        self.models = list()
        for model_set in self.modules.values():
            self.models.extend(model_set)

    def get_model_class(self, supported_framework, module_name, model_name):
        model_class = getattr(import_module(supported_framework), model_name)
        return model_class

    def train(self, model, dataset):
        X = dataset.loc[:, dataset.columns != 'target']
        y = dataset.loc[:, 'target'].ravel()
        model.fit(X, y)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def upload(self, model_registry_path):
        return joblib.load(model_registry_path)

    def save_insidemodel(self, model, mlops_path, saving_name):
        saving_name = saving_name + '.joblib'
        model_registry_saving_path = os.path.join(mlops_path, saving_name)
        joblib.dump(model, model_registry_saving_path)

        training_info_detail = dict()
        training_info_detail['saving_model_name'] = saving_name
        return training_info_detail

    def save_outsidemodel(self, model, model_registry_path, outsidelog_path):
        extension = '.joblib'
        model_registry_path = model_registry_path + extension
        outsidelog = pd.read_csv(outsidelog_path)
        outsidelog.iat[0, 3] = outsidelog.iat[0, 3] + extension
        outsidelog.to_csv(outsidelog_path, index=False)
        return joblib.dump(model, model_registry_path)


class FrameworkCatboost(Framework):
    def __init__(self):
        self.modules = dict()
        self.modules['catboost_model'] = list(filter(lambda x: re.search('Classifier|Regressor', x), import_module('catboost').__all__))

        self.models = list()
        for model_set in self.modules.values():
            self.models.extend(model_set)

    def get_model_class(self, supported_framework, module_name, model_name):
        model_class = getattr(import_module(supported_framework), model_name)
        return model_class

    def train(self, model, dataset):
        X = dataset.loc[:, dataset.columns != 'target']
        y = dataset.loc[:, 'target'].ravel()
        model.fit(X, y)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def upload(self, model_registry_path):
        return joblib.load(model_registry_path)

    def save_insidemodel(self, model, mlops_path, saving_name):
        saving_name = saving_name + '.joblib'
        model_registry_saving_path = os.path.join(mlops_path, saving_name)
        joblib.dump(model, model_registry_saving_path)

        training_info_detail = dict()
        training_info_detail['saving_model_name'] = saving_name
        return training_info_detail

    def save_outsidemodel(self, model, model_registry_path, outsidelog_path):
        extension = '.joblib'
        model_registry_path = model_registry_path + extension
        outsidelog = pd.read_csv(outsidelog_path)
        outsidelog.iat[0, 3] = outsidelog.iat[0, 3] + extension
        outsidelog.to_csv(outsidelog_path, index=False)
        return joblib.dump(model, model_registry_path)


class MLTrigger:
    def __init__(self):
        self.sklearn = FrameworkSklearn()
        self.xgboost = FrameworkXgboost()
        self.lightgbm = FrameworkLightgbm()
        self.catboost = FrameworkCatboost()
        self.supported_frameworks = ['sklearn', 'xgboost', 'lightgbm', 'catboost']

    def preprocessing(self, entry_point=None):
        # User Interfaces
        if isinstance(self._user_datasets, tuple):
            if len(self._user_datasets) == 2 and isinstance(self._user_datasets[1], str):
                # mlops.dataset = (dataset, 'dataset_comment')  >>>  [(dataset, 'dataset_comment')]
                self._user_datasets = [self._user_datasets]
        if not isinstance(self._user_datasets, (list, tuple)):
            # mlops.dataset = dataset  >>>  [dataset]
            self._user_datasets = [self._user_datasets]

        self.preprocessing_information = list()
        for idx_dataset, dataset in enumerate(self._user_datasets):
            if isinstance(dataset, tuple):
                if len(dataset) == 1:
                    # >>> [(dataset0, ), (dataset1, ), (dataset2, ), ...]
                    dataset = dataset[0]
                if len(dataset) == 2:
                    # >>> [(dataset0, 'dataset0_comment'), (dataset1, 'dataset1_comment'), (dataset2, 'dataset2_comment'), ...]
                    dataset, d_comment = dataset
            else:
                # >>> [dataset0, dataset1, dataset2, ...]
                d_comment = None

            saving_time = datetime.today().strftime('%Y%m%d_%H%M%S')
            dataset_name = f'dataset{idx_dataset}.csv'
            dataset_saving_name = saving_time + '-' + dataset_name
            saving_path = os.path.join(self.core['FS'].path, dataset_saving_name)

            dataset.to_csv(saving_path, index=False)
            self.preprocessing_information.append((idx_dataset, dataset_saving_name, saving_time, d_comment, entry_point))
        self._dataset = dataset # last dataset
        return self._dataset.copy()

    def learning(self, entry_point=None):
        # User Interfaces
        if isinstance(self._user_models, tuple):
            if len(self._user_models) == 2 and isinstance(self._user_models[1], str):
                # mlops.model = (model, 'model_comment')  >>>  [(model, 'model_comment')]
                self._user_models = [self._user_models]
        if not isinstance(self._user_models, (list, tuple)):
            # mlops.model = model  >>>  [model]
            self._user_models = [self._user_models]
        
        self.training_information = dict()
        self.training_information['L1'] = list() # for self._user_models
        for idx_model, user_model in enumerate(self._user_models):
            if isinstance(user_model, tuple):
                if len(user_model) == 1:
                    # >>> [(model0, ), (model1, ), (model2, ), ...]
                    user_model = user_model[0]
                if len(user_model) == 2:
                    # >>> [(model0, 'model0_comment'), (model1, 'model1_comment'), (model2, 'model2_comment'), ...]
                    user_model, t_comment = user_model
            else:
                # >>> [model0, model1, model2, ...]
                t_comment = None

            for idx_dataset, dataset in enumerate(self._user_datasets):
                if isinstance(dataset, tuple):
                    if len(dataset) == 1:
                        # >>> [(dataset0, ), (dataset1, ), (dataset2, ), ...]
                        dataset = dataset[0]
                    if len(dataset) == 2:
                        # >>> [(dataset0, 'dataset0_comment'), (dataset1, 'dataset1_comment'), (dataset2, 'dataset2_comment'), ...]
                        dataset, _ = dataset # d_comment 

                _break_l1 = False
                _break_l2 = False
                # Requires optimization on code
                for supported_framework in self.supported_frameworks:
                    for module_name, models in getattr(self, supported_framework).modules.items():
                        for model_name in models:
                            if isinstance(user_model, getattr(self, supported_framework).get_model_class(supported_framework, module_name, model_name)):
                                framework_name = supported_framework
                                framework = getattr(self, framework_name)
                                
                                # training job
                                trainingjob_start_time = datetime.today().strftime('%Y%m%d_%H%M%S')
                                if not entry_point:
                                    model = framework.train(user_model, dataset)
                                else:
                                    model = self.entry_point['train'](user_model, dataset)

                                trainingjob_end_time = datetime.today().strftime('%Y%m%d_%H%M%S')
                                saving_name = trainingjob_end_time + '-' + f'{model_name}'
                                training_info_detail = framework.save_insidemodel(model, mlops_path=self.core['MR'].path, saving_name=model_name)
                                training_info_detail['training_start_time'] = trainingjob_start_time
                                training_info_detail['training_end_time'] = trainingjob_end_time
                                training_info_detail['t_comment'] = t_comment
                                _break_l1 = True
                                break
                        if _break_l1:
                            _break_l2 = True
                            break
                    if _break_l2:
                        self._dataset_idx = idx_dataset
                        self._dataset = dataset
                        self._model = model
                        self._model_name = model_name
                        self._model_idx = idx_model
                        self._framework = framework
                        self._framework_name = framework_name
                        self._training_info_detail = training_info_detail
                        break
                self.training_information['L1'].append((
                    self._model_idx, self._dataset_idx, self._model_name, self._framework_name, 
                    deepcopy(self._model), self._framework, 
                    self._training_info_detail['training_start_time'], 
                    self._training_info_detail['training_end_time'], 
                    self._training_info_detail['saving_model_name'],
                    self._training_info_detail['t_comment'],
                    ))

        info_train = pd.DataFrame(
                data=list(map(lambda x: (x[0], x[1], x[2], x[3], x[6], x[7], x[8], x[9]), self.training_information['L1'])), 
                columns=self._insidelog_entities['train'])
        info_dataset = pd.DataFrame(
                data=self.preprocessing_information, 
                columns=self._insidelog_entities['preprocessing'])

        self.inside_board = pd.merge(info_train, info_dataset, how='left', on='d_idx')
        self.inside_board['from'] = 'inside'
        mlops_log = self.inside_board.copy()

        logging_history = list(filter(lambda x: re.search(self._insidelog_name[:-4], x), self.core['MS'].listdir()))
        logging_path = os.path.join(self.core['MS'].path, self._insidelog_name)
        if bool(len(logging_history)):
            mlops_log = mlops_log.append(pd.read_csv(logging_path), ignore_index=True)
        self.insidelog = mlops_log
        self.insidelog.to_csv(logging_path, index=False)
        self._model = self.training_information['L1'][-1][4] # last model
        return deepcopy(self._model)

    def prediction(self, X=None):
        if X is None:
            X = self._dataset.loc[:, self._dataset.columns != 'target']
        elif isinstance(X, slice):
            X = self._dataset.loc[X, self._dataset.columns != 'target']
        return self._framework.predict(self._model, X)

    def get_model(self, model_registry_path):
        return self._framework.upload(model_registry_path)
    
    def put_model(self, model, model_registry_path):
        outsidelog_path = os.path.join(self.core['MS'].path, self._outsidelog_name)
        return self._framework.save_outsidemodel(model, model_registry_path, outsidelog_path)


class MLOps(MLTrigger):
    def __init__(self, mlops_bs):
        super(MLOps, self).__init__()
        self.core = mlops_bs.core
        self._entry_point = False
        self.__dataset = None
        self.__model = None
        
        # usage with the 'inside_board' definition inside the 'def learning' on the class 'MLTrigger'
        # self.inside_board > self.insidelog
        self._insidelog_name = 'mlops_insidelog.csv'
        self._insidelog_entities = dict() # columns of insidelog
        self._insidelog_entities['preprocessing'] = ['d_idx', 'd_saving_name', 'd_saving_time', 'd_comment'] + ['c_entry_point']
        self._insidelog_entities['train'] = ['t_idx', 'd_idx', 'model_name', 'framework_name', 't_start_time', 't_end_time', 't_saving_name', 't_comment']
        self._insidelog_entities['logtype'] = ['from']
        self._insidelog_columns = self._insidelog_entities['train'] + self._insidelog_entities['preprocessing'][1:] + self._insidelog_entities['logtype']
        
        # usage with the 'appending_board' definition inside the 'def storing_model'
        self._outsidelog_name = 'mlops_outsidelog.csv'
        self._outsidelog_columns = ['t_idx', 'model_name', 'framework_name', 't_saving_name', 'from', 'comment']

        # usage with the 'self.commitlog' definition inside the 'def codecommit'
        self._commitlog_name = 'mlops_commitlog.csv'
        self._commitlog_columns = self._insidelog_columns
        
        # insidelog
        logging_history = list(filter(lambda x: re.search(self._insidelog_name[:-4], x), self.core['MS'].listdir()))
        logging_path = os.path.join(self.core['MS'].path, self._insidelog_name)
        if bool(len(logging_history)):
            self.insidelog = pd.read_csv(logging_path)
        else:
            self.insidelog = pd.DataFrame(columns=self._insidelog_columns)
        
        # outsidelog
        logging_history = list(filter(lambda x: re.search(self._outsidelog_name[:-4], x), self.core['MS'].listdir()))
        logging_path = os.path.join(self.core['MS'].path, self._outsidelog_name)
        if bool(len(logging_history)):
            self.outsidelog = pd.read_csv(logging_path)
        else:
            self.outsidelog = pd.DataFrame(columns=self._outsidelog_columns)
        
        # commitlog
        logging_history = list(filter(lambda x: re.search(self._commitlog_name[:-4], x), self.core['MS'].listdir()))
        logging_path = os.path.join(self.core['MS'].path, self._commitlog_name)
        if bool(len(logging_history)):
            self.commitlog = pd.read_csv(logging_path)
        else:
            self.commitlog = pd.DataFrame(columns=self._insidelog_columns)

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, datasets):
        self._user_datasets = datasets
        self.__dataset = self.preprocessing()
    
    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, models):
        self._user_models = models
        self.__model = self.learning()
 
    def inference(self, X=None):
        return self.prediction(X)
    
    def evaluation(self):
        return 

    def training_board(self, log=None):
        if not log:
            # D[self._user_datasets] X D[self._user_models]
            return self.inside_board
        elif log == 'inside':
            # All history through mlops.dataset, mlops.model
            return self.insidelog
        elif log == 'outside':
            # All history through mlops.storing_model
            return self.outsidelog
        elif log == 'commit':
            return self.commitlog
        else:
            return self.inside_board

    def feature_choice(self, idx:int=-1):
        if isinstance(idx, int):
            self._dataset_num = len(self._user_datasets)
            if idx == -1:
                self._dataset_idx = self._dataset_num + idx
            else:
                self._dataset_idx = idx
            self._dataset = self._user_datasets[idx]
            self.__dataset = self._dataset.copy()

        elif isinstance(idx, str):
            saving_dataset_name = idx
            dataset_path = os.path.join(self.core['FS'].path, saving_dataset_name)
            self._dataset = pd.read_csv(dataset_path)
            self.__dataset = self._dataset.copy()
        return self

    def model_choice(self, idx:int=-1):
        if isinstance(idx, int):
            self._model_num = len(self._user_models)
            if idx == -1:
                self._model_idx = self._model_num + idx
            else:
                self._model_idx = idx
            self._framework_name = self.inside_board.loc[lambda x: (x.t_idx == idx)&(x.d_idx == self._dataset_idx), 'framework_name'].item()
            self._framework = getattr(self, self._framework_name)
            self._training_info_detail = {
                    'training_start_time': self.inside_board.loc[lambda x: (x.t_idx == idx)&(x.d_idx == self._dataset_idx), 't_start_time'],
                    'training_end_time': self.inside_board.loc[lambda x: (x.t_idx == idx)&(x.d_idx == self._dataset_idx), 't_end_time'],
                    'saving_model_name': self.inside_board.loc[lambda x: (x.t_idx == idx)&(x.d_idx == self._dataset_idx), 't_saving_name']
                    }
            self._model = self.training_information['L1'][self._dataset_num*(idx) + self._dataset_idx][4]
            self.__model = deepcopy(self._model)
        elif isinstance(idx, str):
            saving_model_name = idx
            insidelog_frame = self.insidelog.loc[lambda x: x.t_saving_name == saving_model_name]
            outsidelog_frame = self.outsidelog.loc[lambda x: x.t_saving_name == saving_model_name]
            if insidelog_frame.shape[0] == 1:
                self._framework_name = insidelog_frame['framework_name'].item()
            elif outsidelog_frame.shape[0] == 1:
                self._framework_name = outsidelog_frame['framework_name'].item()
            self._framework = getattr(self, self._framework_name)
            model_path = os.path.join(self.core['MR'].path, saving_model_name)
            self._model = self.get_model(model_path)
            self.__model = deepcopy(self._model)
        else:
            pass
        return self

    def drawup_dataset(self, name):
        return pd.read_csv(os.path.join(self.core['FS'].path, name))

    def drawup_model(self, name):
        insidelog_frame = self.insidelog.loc[lambda x: x.t_saving_name == name]
        outsidelog_frame = self.outsidelog.loc[lambda x: x.t_saving_name == name]
        if insidelog_frame.shape[0] == 1:
            self._framework_name = insidelog_frame['framework_name'].item()
        elif outsidelog_frame.shape[0] == 1:
            self._framework_name = outsidelog_frame['framework_name'].item()
        else:
            print('Not matched!')
            return None
        self._framework = getattr(self, self._framework_name)
        model_path = os.path.join(self.core['MR'].path, name)
        return self.get_model(model_path)
 
    def drawup_source(self, name):
        return copyfile(os.path.join(self.core['SR'].path, name), name)

    def display_source(self, name):
        source_code_in_source_repository = os.path.join(self.core['SR'].path, name)
        with open(source_code_in_source_repository, 'r') as f:
            print(f.read())

    def storing_model(self, user_model, comment=None):
        framework = None
        framework_name = None
        for supported_framework in self.supported_frameworks:
            for module_name, models in getattr(self, supported_framework).modules.items():
                for model_name in models:
                    if isinstance(user_model, getattr(self, supported_framework).get_model_class(supported_framework, module_name, model_name)):
                        framework_name = supported_framework
                        framework = getattr(self, framework_name)
                        model_name = model_name
        if not framework_name:
            return
        else:
            self._framework_name = framework_name
            self._framework = framework
            self._model = user_model

        # saving model with refreshing outsidelog
        saving_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        model_registry_path = os.path.join(self.core['MR'].path, saving_time + '-' + model_name)

        appending_board = pd.DataFrame(
                columns=self._outsidelog_columns,
                data=[[self.outsidelog.shape[0], model_name, self._framework_name, saving_time + '-' + model_name, 'outside', comment]])
        self.outsidelog = appending_board.append(self.outsidelog, ignore_index=True)
        self.outsidelog.to_csv(os.path.join(self.core['MS'].path, self._outsidelog_name), index=False)
        self.put_model(user_model, model_registry_path)
        self.outsidelog = pd.read_csv(os.path.join(self.core['MS'].path, self._outsidelog_name))

    def codecommit(self, entry_point):
        entry_name = entry_point[:-3] # *.py

        self.entry_point = dict()
        self.entry_point['source'] = import_module(entry_name)
        mlops_entry_point = datetime.today().strftime('%Y%m%d_%H%M%S') + '-' + entry_point
        if hasattr(self.entry_point['source'], 'preprocessing'):
            self.entry_point['preprocessing'] = getattr(import_module(entry_name), 'preprocessing')  # return datasets
            self._user_datasets = self.entry_point['preprocessing']()
            self.__dataset = self.preprocessing(entry_point=mlops_entry_point)
        if hasattr(self.entry_point['source'], 'architecture'):
            self.entry_point['architecture'] = getattr(self.entry_point['source'], 'architecture')   # return user_models
            self._user_models = self.entry_point['architecture']()
        if hasattr(self.entry_point['source'], 'train'):
            self.entry_point['train'] = getattr(import_module(entry_name), 'train')                  # return model
            self.__model = self.learning(entry_point=mlops_entry_point)
        if hasattr(self.entry_point['source'], 'predict'):
            self.entry_point['predict'] = getattr(import_module(entry_name), 'predict')
        if hasattr(self.entry_point['source'], 'evaluate'):
            self.entry_point['evaluate'] = getattr(import_module(entry_name), 'evaluate')            # return metrics
        if hasattr(self.entry_point['source'], 'report'):
            self.entry_point['report'] = getattr(import_module(entry_name), 'report')                # return report
    
        copyfile(entry_point, os.path.join(self.core['SR'].path, mlops_entry_point))
        self.commitlog = self.insidelog.loc[self.insidelog['c_entry_point'].dropna().index].reset_index(drop=True)
        self.commitlog.to_csv(os.path.join(self.core['MS'].path, self._commitlog_name), index=False)

    def summary(self):
        return
