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

    def train(self, model, dataset, mlops_path):
        model.fit(dataset.loc[:, dataset.columns != 'target'], dataset.loc[:, 'target'])
        joblib.dump(model, mlops_path)
        return model

    def upload(self):
        return


class FrameworkXgboost(Framework):
    def __init__(self):
        self.modules = dict()

    def train(self, model, dataset, mlops_path):
        model.fit(dataset.loc[:, dataset.columns != 'target'], dataset.loc[:, 'target'])

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
        
        metric_data = list()
        self._fitted_models = list()
        for user_model in self._user_models:
            _break = False
            for module_name, models in self.sklearn.modules.items():
                for model_name in models:
                    if isinstance(user_model, getattr(getattr(sklearn, module_name), model_name)):
                        framework_name = 'sklearn'
                        model_name = model_name

                        framework = getattr(self, framework_name)
                        model_registry_path = os.path.join(self.core['MR'].path, datetime.today().strftime('%Y%m%d-%H%M%S-') + f'{model_name}.joblib')
                        model = framework.train(user_model, self._dataset, model_registry_path)
                        self._fitted_models.append(model)

                        _break = True
                        break

                #metadata_store_path = os.path.join(self.core['MS'].path, datetime.today().strftime('%Y%m%d-%H%M%S-') + f'{model_name}.joblib')
                #score = framework.predict(self._model, self._dataset, metadata_store_path)
                #metric_data.append((framework_name.upper(), model_name.upper(), score))

                if _break:
                    self._model = model
                    break
                else:
                    continue

        #metric_report = pd.DataFrame(metric_data, columns=['FrameWork', 'Model', 'Score'])
        #best_model = fitted_models[metric_report.Score.argmax()]
        self._model = self._fitted_models[0]
        return self._model


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
        return self.__model.predict(X)
    
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
