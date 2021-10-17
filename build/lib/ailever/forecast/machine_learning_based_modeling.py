from ailever.forecast import __forecast_bs__ as forecast_bs

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import linear_model, tree, neighbors, svm, ensemble
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import pandas as pd

model = dict()
model['LR'] = linear_model.LinearRegression()
model['LASSO'] = linear_model.Lasso()
model['EN'] = linear_model.ElasticNet()
model['KNN'] = neighbors.KNeighborsRegressor()
model['CART'] = tree.DecisionTreeRegressor()
model['SVR'] = svm.SVR()
model['AB'] = ensemble.AdaBoostRegressor()
model['GBM'] = ensemble.GradientBoostingRegressor()
model['RF'] = ensemble.RandomForestRegressor()
model['ET'] = ensemble.ExtraTreesRegressor()

class LightMLOps:
    def __init__(self, dataset, target):
        # Dataset
        num_instance = dataset[target].unique().shape[0]
        self.problem = 'classification' if num_instance == 2 else 'regression'
            
        self.dataset_X = dataset.drop(target, axis=1).values
        self.dataset_Y = dataset[target].values
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(self.dataset_X, self.dataset_Y, test_size=0.2)
        
        # Models
        self.regressors = ['SVR', 'GBM']
        self.classifiers = [] 
        self.training_models = self.classifiers if self.problem == 'classification' else self.regressors
        
        self.baselines = list()
        self.pipelines = list()
        for training_model in self.training_models:
            # Baseline Model
            self.baselines.append((training_model, model[training_model]))
            # Scaled Model
            self.pipelines.append(('Scaled'+training_model, Pipeline([('Scaler', StandardScaler()),(training_model, model[training_model])])))        
        
    def __getitem__(self):
        pass
    
    def trigger(self):
        self.spot_check_block()
        self.scaling_block()
        self.tunning_block()
        
    def spot_check_block(self):
        results = []
        names = []
        for name, model in self.baselines:
            scoring = 'neg_mean_squared_error'
            kfold = KFold(n_splits=10, shuffle=True)
            cv_results = cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

        # Compare Algorithms
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
    
    def scaling_block(self):
        results = []
        names = []
        for name, model in self.pipelines:
            scoring = 'neg_mean_squared_error'
            kfold = KFold(n_splits=10, shuffle=True)
            cv_results = cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

        # Compare Algorithms
        fig = plt.figure()
        fig.suptitle('Scaled Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
    
    def tunning_block(self):
        # Tune scaled SVM
        scaler = StandardScaler().fit(self.X_train)
        rescaledX = scaler.transform(self.X_train)
        c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
        kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
        param_grid = dict(C=c_values, kernel=kernel_values)

        for name, model in self.baselines:
            if name == 'SVR':
                scoring = 'neg_mean_squared_error'
                kfold = KFold(n_splits=10, shuffle=True)
                grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
                grid_result = grid.fit(rescaledX, self.Y_train)
                print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                means = grid_result.cv_results_['mean_test_score']
                stds = grid_result.cv_results_['std_test_score']
                params = grid_result.cv_results_['params']
                for mean, stdev, param in zip(means, stds, params):
                    print("%f (%f) with: %r" % (mean, stdev, param))
    
    def ensemble_block(self):
        pass
    
    def feature_store(self):
        pass

    def model_registry(self):
        pass

    def analysis_report_repository(self):
        pass
    


