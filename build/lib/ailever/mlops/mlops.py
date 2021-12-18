from abc import *
import os
from datetime import datetime
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

class FrameworkSklearn(Framework):
    def __init__(self):
        self.modules = dict()
        self.modules['linear_model'] = list(filter(lambda x: x[-10:] == 'Classifier' or x[-9:] == 'Regressor', sklearn.linear_model.__all__))
        self.modules['ensemble'] = list(filter(lambda x: x[-10:] == 'Classifier' or x[-9:] == 'Regressor', sklearn.ensemble.__all__))
        self.modules['neighbors'] = list(filter(lambda x: x[-10:] == 'Classifier' or x[-9:] == 'Regressor', sklearn.neighbors.__all__))
        self.modules['tree'] = list(filter(lambda x: x[-10:] == 'Classifier' or x[-9:] == 'Regressor', sklearn.tree.__all__))
        self.modules['svm'] = list(filter(lambda x: x[-3:] == 'SVC' or x[-3:] == 'SVR', sklearn.svm.__all__))

    def train(self, model, dataset):
        model.fit(dataset[dataset.columns != 'target'], dataset['target'])
        saving_path = os.path.join(self.core['MR'].path, datetime.today().strftime('%Y%m%d-%H%M%S-') + f'{model_name}.joblib')
        joblib.dump(model, saving_path)
        return model

class FrameworkXgboost(Framework):
    def __init__(self):
        self.modules = dict()

    def train(self, model, dataset):
        model.fit(dataset[dataset.columns != 'target'], dataset['target'])



class AutoML:
    def __init__(self):
        self.sklearn = FrameworkSklearn()
        self.xgboost = FrameworkXgboost()

    def preprocessing(self):
        saving_path = os.path.join(self.core['FS'].path, datetime.today().strftime('%Y%m%d-%H%M%S-') + 'dataset.csv')
        self._dataset.to_csv(saving_path, index=False)
        return self._dataset

    def learning(self):
        for module, models in self.sklearn.modules.item():
            #if any([isinstance(self._model, getattr(sklearn, module)) for model in models]):
            for model in models:
                if isinstance(self._model, getattr(sklearn, module))
                    frmaework_name = 'sklearn'
                    model_name = model
                    _break = True
                    break
            if _break:
                break
            else:
                continue

        framework = getattr(self, framework_name)
        model = framework.train(self._model, self._dataset)
        return model



class MLOps(AutoML):
    def __init__(self, mlops_bs):
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
    def model(self, model):
        self._model = model
        self.__model = self.learning()

    def predict(self):
        pass
    
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
