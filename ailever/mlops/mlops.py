from abc import *
import os
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
        self.modules['linear_model'] = list(filter(lambda x: x[-10:] == 'Classifier' or x[-9:] == 'Regressor', sklearn.linear_model.__all__))
        self.modules['ensemble'] = list(filter(lambda x: x[-10:] == 'Classifier' or x[-9:] == 'Regressor', sklearn.ensemble.__all__))
        self.modules['neighbors'] = list(filter(lambda x: x[-10:] == 'Classifier' or x[-9:] == 'Regressor', sklearn.neighbors.__all__))
        self.modules['tree'] = list(filter(lambda x: x[-10:] == 'Classifier' or x[-9:] == 'Regressor', sklearn.tree.__all__))
        self.modules['svm'] = list(filter(lambda x: x[-3:] == 'SVC' or x[-3:] == 'SVR', sklearn.svm.__all__))

    def train(self, model, dataset, mlops_path, saving_name):
        model_registry_path = os.path.join(mlops_path, datetime.today().strftime('%Y%m%d-%H%M%S-') + f'{saving_name}.joblib')
        model.fit(dataset.loc[:, dataset.columns != 'target'], dataset.loc[:, 'target'])
        joblib.dump(model, model_registry_path)
        return model

    def predict(self):
        return 

    def upload(self):
        return


class FrameworkXgboost(Framework):
    def __init__(self):
        self.modules = dict()

    def train(self, model, dataset, mlops_path, saving_name):
        model_registry_path = os.path.join(mlops_path, datetime.today().strftime('%Y%m%d-%H%M%S-') + f'{saving_name}.joblib')
        model.fit(dataset.loc[:, dataset.columns != 'target'], dataset.loc[:, 'target'])
        joblib.dump(model, model_registry_path)
        return model

    def predict(self):
        pass

    def upload(self):
        return


class AutoML:
    def __init__(self):
        self.sklearn = FrameworkSklearn()
        self.xgboost = FrameworkXgboost()

    def preprocessing(self):
        saving_path = os.path.join(self.core['FS'].path, datetime.today().strftime('%Y%m%d-%H%M%S-') + 'dataset.csv')
        self._dataset.to_csv(saving_path, index=False)
        return self._dataset

    def learning(self):
        if not isinstance(self._user_models, list):
            self._user_models = [self._user_models]
        
        self.training_information = list()
        for idx, user_model in enumerate(self._user_models):
            _break = False
            for module_name, models in self.sklearn.modules.items():
                for model_name in models:
                    if isinstance(user_model, getattr(getattr(sklearn, module_name), model_name)):
                        framework = getattr(self, 'sklearn')
                        model = framework.train(user_model, self._dataset, mlops_path=self.core['MR'].path, saving_name=model_name)

                        _break = True
                        break
                if _break:
                    self._model = model
                    self._framework = framework
                    break
                else:
                    continue

            self.training_information.append((idx, model_name, self._framework, self._model))
        self._model = self.training_information[0][-1]
        return self._model

    def prediction(self, X):
        return self.training_information[0][2].predict(X)


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
