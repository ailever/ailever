from abc import *
from importlib import import_module
import os
import re
from datetime import datetime
from copy import deepcopy
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
    def save_outsidemodel(self):
        pass


class FrameworkSklearn(Framework):
    def __init__(self):
        self.modules = dict()
        self.modules['linear_model'] = list(filter(
            lambda x: re.search('Classifier|Regression|Regressor', x), import_lib('sklearn.linear_model').__all__))
        self.modules['ensemble'] = list(filter(
            lambda x: re.search('Classifier|Regressor', x), import_lib('sklearn.ensemble').__all__))
        self.modules['neighbors'] = list(filter(
            lambda x: re.search('Classifier|Regressor', x), import_lib('sklearn.neighbors').__all__))
        self.modules['tree'] = list(filter(
            lambda x: re.search('Classifier|Regressor', x), import_lib('sklearn.tree').__all__))
        self.modules['svm'] = list(filter(
            lambda x: re.search('SVC|SVR', x), import_lib('sklearn.svm').__all__))
 
        self.models = list()
        for model_set in self.modules.values():
            self.models.extend(model_set)

    def get_model_class(self, supported_framework, module_name, model_name):
        model_class = getattr(getattr(globals()[supported_framework], module_name), model_name)
        return model_class

    def train(self, model, dataset, mlops_path, saving_name):
        training_info_detail = dict()
        training_start_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        X = dataset.loc[:, dataset.columns != 'target']
        y = dataset.loc[:, 'target'].ravel()
        model.fit(X, y)
        training_end_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        
        saving_name = training_end_time + '-' + f'{saving_name}.joblib'
        model_registry_path = os.path.join(mlops_path, saving_name)
        joblib.dump(model, model_registry_path)
        training_info_detail['training_start_time'] = training_start_time
        training_info_detail['training_end_time'] = training_end_time
        training_info_detail['saving_model_name'] = saving_name
        return model, training_info_detail

    def predict(self, model, X):
        return model.predict(X)

    def upload(self, model_registry_path):
        return joblib.load(model_registry_path)

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
        self.modules['xgboost_model'] = list(filter(lambda x: re.search('Classifier|Regressor', x), import_lib('xgboost').__all__))

        self.models = list()
        for model_set in self.modules.values():
            self.models.extend(model_set)

    def get_model_class(self, supported_framework, module_name, model_name):
        model_class = getattr(globals()[supported_framework], model_name)
        return model_class

    def train(self, model, dataset, mlops_path, saving_name):
        training_info_detail = dict()
        training_start_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        X = dataset.loc[:, dataset.columns != 'target']
        y = dataset.loc[:, 'target'].ravel()
        model.fit(X, y)
        training_end_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        
        saving_name = training_end_time + '-' + f'{saving_name}.joblib'
        model_registry_path = os.path.join(mlops_path, saving_name)
        joblib.dump(model, model_registry_path)
        training_info_detail['training_start_time'] = training_start_time
        training_info_detail['training_end_time'] = training_end_time
        training_info_detail['saving_model_name'] = saving_name
        return model, training_info_detail

    def predict(self, model, X):
        return model.predict(X)

    def upload(self, model_registry_path):
        return joblib.load(model_registry_path)

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
        self.modules['lightgbm_model'] = list(filter(lambda x: re.search('Classifier|Regressor', x), import_lib('lightgbm').__all__))

        self.models = list()
        for model_set in self.modules.values():
            self.models.extend(model_set)

    def get_model_class(self, supported_framework, module_name, model_name):
        model_class = getattr(globals()[supported_framework], model_name)
        return model_class

    def train(self, model, dataset, mlops_path, saving_name):
        training_info_detail = dict()
        training_start_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        X = dataset.loc[:, dataset.columns != 'target']
        y = dataset.loc[:, 'target'].ravel()
        model.fit(X, y)
        training_end_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        
        saving_name = training_end_time + '-' + f'{saving_name}.joblib'
        model_registry_path = os.path.join(mlops_path, saving_name)
        joblib.dump(model, model_registry_path)
        training_info_detail['training_start_time'] = training_start_time
        training_info_detail['training_end_time'] = training_end_time
        training_info_detail['saving_model_name'] = saving_name
        return model, training_info_detail

    def predict(self, model, X):
        return model.predict(X)

    def upload(self, model_registry_path):
        return joblib.load(model_registry_path)

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
        self.modules['catboost_model'] = list(filter(lambda x: re.search('Classifier|Regressor', x), import_lib('catboost').__all__))

        self.models = list()
        for model_set in self.modules.values():
            self.models.extend(model_set)

    def get_model_class(self, supported_framework, module_name, model_name):
        model_class = getattr(globals()[supported_framework], model_name)
        return model_class

    def train(self, model, dataset, mlops_path, saving_name):
        training_info_detail = dict()
        training_start_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        X = dataset.loc[:, dataset.columns != 'target']
        y = dataset.loc[:, 'target'].ravel()
        model.fit(X, y)
        training_end_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        
        saving_name = training_end_time + '-' + f'{saving_name}.joblib'
        model_registry_path = os.path.join(mlops_path, saving_name)
        joblib.dump(model, model_registry_path)
        training_info_detail['training_start_time'] = training_start_time
        training_info_detail['training_end_time'] = training_end_time
        training_info_detail['saving_model_name'] = saving_name
        return model, training_info_detail

    def predict(self, model, X):
        return model.predict(X)

    def upload(self, model_registry_path):
        return joblib.load(model_registry_path)

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

    def preprocessing(self):
        if not isinstance(self._user_datasets, list):
            self._user_datasets = [self._user_datasets]

        self.preprocessing_information = list()
        for idx_dataset, dataset in enumerate(self._user_datasets):
            dataset_name = f'dataset{idx_dataset}.csv'
            saving_time = datetime.today().strftime('%Y%m%d_%H%M%S')
            dataset_saving_name = saving_time + '-' + dataset_name
            saving_path = os.path.join(self.core['FS'].path, dataset_saving_name)
            
            dataset.to_csv(saving_path, index=False)
            self.preprocessing_information.append((idx_dataset, dataset_saving_name, saving_time))


    def learning(self):
        if not isinstance(self._user_models, list):
            self._user_models = [self._user_models]
        
        self.training_information = dict()
        self.training_information['L1'] = list() # for self._user_models
        for idx_model, user_model in enumerate(self._user_models):
            for idx_dataset, dataset in enumerate(self._user_datasets):
                _break_l1 = False
                _break_l2 = False
                # Requires optimization on code
                for supported_framework in self.supported_frameworks:
                    for module_name, models in getattr(self, supported_framework).modules.items():
                        for model_name in models:
                            if isinstance(user_model, getattr(self, supported_framework).get_model_class(supported_framework, module_name, model_name)):
                                framework_name = supported_framework
                                framework = getattr(self, framework_name)
                                model, training_info_detail = framework.train(user_model, dataset, mlops_path=self.core['MR'].path, saving_name=model_name)
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
                    self._model_idx, self._dataset_idx, self._model_name, self._framework_name, deepcopy(self._model), self._framework,
                    self._training_info_detail['training_start_time'], self._training_info_detail['training_end_time'], self._training_info_detail['saving_model_name']))

        info_train = pd.DataFrame(list(map(lambda x: (x[0], x[1], x[2], x[3], x[6], x[7], x[8]), self.training_information['L1'])), 
                columns=['t_idx', 'd_idx', 'model_name', 'framework_name', 't_start_time', 't_end_time', 't_saving_name'])
        info_dataset = pd.DataFrame(self.preprocessing_information, columns=['d_idx', 'd_saving_name', 'd_saving_time'])

        self.inside_board = pd.merge(info_train, info_dataset, how='left', on='d_idx')
        self.inside_board['from'] = 'inside'
        mlops_log = self.inside_board.copy()

        logging_history = list(filter(lambda x: re.search(self._insidelog_name[:-4], x), self.core['MS'].listdir()))
        logging_path = os.path.join(self.core['MS'].path, self._insidelog_name)
        if bool(len(logging_history)):
            mlops_log = mlops_log.append(pd.read_csv(logging_path), ignore_index=True)
        self.insidelog = mlops_log
        self.insidelog.to_csv(logging_path, index=False)
        self._model = self.training_information['L1'][0][4]
        return self._model

    def prediction(self, X):
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
        self._insidelog_name = 'mlops_insidelog.csv'
        self._outsidelog_name = 'mlops_outsidelog.csv'
        self.__dataset = None
        self.__model = None
    
        logging_history = list(filter(lambda x: re.search(self._insidelog_name[:-4], x), self.core['MS'].listdir()))
        logging_path = os.path.join(self.core['MS'].path, self._insidelog_name)
        if bool(len(logging_history)):
            self.insidelog = pd.read_csv(logging_path)
        else:
            self.insidelog = pd.DataFrame(columns=['t_idx', 'd_idx', 'model_name', 'framwork_name', 't_start_time', 't_end_time', 't_saving_name', 'd_saving_name', 'd_saving_time', 'from'])

        logging_history = list(filter(lambda x: re.search(self._outsidelog_name[:-4], x), self.core['MS'].listdir()))
        logging_path = os.path.join(self.core['MS'].path, self._outsidelog_name)
        if bool(len(logging_history)):
            self.outsidelog = pd.read_csv(logging_path)
        else:
            self.outsidelog = pd.DataFrame(columns=['t_idx', 'model_name', 'framework_name', 't_saving_name', 'from', 'comment'])

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
        
    def inference(self, X):
        return self.prediction(X)
    
    def training_board(self, log=None):
        if not log:
            return self.inside_board
        elif log == 'inside':
            return self.insidelog
        elif log == 'outside':
            return self.outsidelog
        else:
            return self.inside_board

    def feature_choice(self, idx):
        self._dataset_idx = idx
        self._dataset_num = len(self._user_datasets)
        self._dataset = self._user_datasets[idx]
        self.__dataset = self._dataset.copy()
        return self

    def model_choice(self, idx):
        if isinstance(idx, int):
            self._framework_name = self.inside_board.loc[lambda x: (x.t_idx == idx)&(x.d_idx == self._dataset_idx), 'framework_name'].item()
            self._framework = getattr(self, self._framework_name)
            self._model_idx = idx
            self._model_num = len(self._user_models)
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
 
    def storing_model(self, user_model, comment='-'):
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
                columns=['t_idx',                 'framework_name', 'model_name', 't_saving_name',                'from', 'comment'],
                data=  [[self.outsidelog.shape[0], self._framework_name, model_name,  saving_time + '-' + model_name, 'outside', comment]])
        self.outsidelog = appending_board.append(self.outsidelog, ignore_index=True)
        self.outsidelog.to_csv(os.path.join(self.core['MS'].path, self._outsidelog_name), index=False)
        self.put_model(user_model, model_registry_path)
        self.outsidelog = pd.read_csv(os.path.join(self.core['MS'].path, self._outsidelog_name))

    def codecommit(self, entry_point):
        preprocessing = getattr(import_module(entry_point), 'preprocessing')
        train = getattr(import_module(entry_point), 'train')
        predict = getattr(import_module(entry_point), 'predict')

    def summary(self):
        return
