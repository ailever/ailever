from abc import *
import os
import re
from datetime import datetime
import pandas as pd
import sklearn
import xgboost
import joblib

class Framework(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def module_class(self):
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


class FrameworkSklearn(Framework):
    def __init__(self):
        self.modules = dict()
        self.modules['linear_model'] = list(filter(
            lambda x: re.search('Classifier|Regression|Regressor', x), 
            sklearn.linear_model.__all__))
        self.modules['ensemble'] = list(filter(
            lambda x: re.search('Classifier|Regressor', x), 
            sklearn.ensemble.__all__))
        self.modules['neighbors'] = list(filter(
            lambda x: re.search('Classifier|Regressor', x),
            sklearn.neighbors.__all__))
        self.modules['tree'] = list(filter(
            lambda x: re.search('Classifier|Regressor', x), 
            sklearn.tree.__all__))
        self.modules['svm'] = list(filter(
            lambda x: re.search('SVC|SVR', x), 
            sklearn.svm.__all__))
    
    def get_model_class(self, supported_framework, module_name, model_name):
        model_class = getattr(getattr(globals()[supported_framework], module_name), model_name)
        return model_class

    def train(self, model, dataset, mlops_path, saving_name):
        model_registry_path = os.path.join(mlops_path, datetime.today().strftime('%Y%m%d-%H%M%S-') + f'{saving_name}.joblib')
        model.fit(dataset.loc[:, dataset.columns != 'target'], dataset.loc[:, 'target'])
        joblib.dump(model, model_registry_path)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def upload(self):
        return


class FrameworkXgboost(Framework):
    def __init__(self):
        self.modules = dict()
        self.modules['xgboost_model'] = list(filter(lambda x: re.search('Classifier|Regressor', x), xgboost.__all__))

    def get_model_class(self, supported_framework, module_name, model_name):
        model_class = getattr(globals()[supported_framework], model_name)
        return model_class

    def train(self, model, dataset, mlops_path, saving_name):
        model_registry_path = os.path.join(mlops_path, datetime.today().strftime('%Y%m%d-%H%M%S-') + f'{saving_name}.joblib')
        model.fit(dataset.loc[:, dataset.columns != 'target'], dataset.loc[:, 'target'])
        joblib.dump(model, model_registry_path)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def upload(self):
        return


class AutoML:
    def __init__(self):
        self.sklearn = FrameworkSklearn()
        self.xgboost = FrameworkXgboost()
        self.supported_frameworks = ['sklearn', 'xgboost']

    def preprocessing(self):
        saving_path = os.path.join(self.core['FS'].path, datetime.today().strftime('%Y%m%d-%H%M%S-') + 'dataset.csv')
        self._dataset.to_csv(saving_path, index=False)
        return self._dataset

    def learning(self):
        if not isinstance(self._user_models, list):
            self._user_models = [self._user_models]
        
        self.training_information = dict()
        self.training_information['L1'] = list() # for self._user_models
        self.training_information['L2'] = list() # for self._user_models
        for idx, user_model in enumerate(self._user_models):
            _break_l1 = False
            _break_l2 = False
            for supported_framework in self.supported_frameworks:
                for module_name, models in getattr(self, supported_framework).modules.items():
                    for model_name in models:
                        if isinstance(user_model, self.get_model_class(supported_framework, module_name, model_name)):
                            framework = getattr(self, supported_framework)
                            model = framework.train(user_model, self._dataset, mlops_path=self.core['MR'].path, saving_name=model_name)
                            _break_l1 = True
                            break
                    if _break_l1:
                        break_l2 = True
                        break
                if _break_l2:
                    self._model = model
                    self._framework = framework
                    break


            self.training_information['L1'].append((idx, model_name, self._framework, self._model))
        self._model = self.training_information['L1'][0][-1]
        return self._model

    def prediction(self, dataset):
        framework = self.training_information['L1'][0][2]
        model = self.training_information['L1'][0][3]
        return framework.predict(model, dataset)


class MLOps(AutoML):
    def __init__(self, mlops_bs):
        super(MLOps, self).__init__()
        self.core = mlops_bs.core
        self.__dataset = None
        self.__model = None
    
    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset
        self.__dataset = self.preprocessing()
    
    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, user_models):
        self._user_models = user_models
        self.__model = self.learning()

    def inference(self, X):
        self._model = self.__model
        return self.prediction(X)
    
    def feature_choice(self):
        return self

    def model_choice(self):
        return self

    def get_dataset(self):
        return

    def get_model(self):
        return
    
    def summary(self):
        return
