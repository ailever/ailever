import pandas as pd

class Sklearn_API:
    def boston(self, download=False):
        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)
        df = pd.DataFrame(data=X)
        df['target'] = y
        if download:
            df.to_csv('boston.csv')
        return df

    def breast_cancer(self, download=False):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        df = pd.DataFrame(data=X)
        df['target'] = y
        if download:
            df.to_csv('breast_cancer.csv')
        return df

    def diabetes(self, download=False):
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(return_X_y=True)
        df = pd.DataFrame(data=X)
        df['target'] = y
        if download:
            df.to_csv('diabetes.csv')
        return df

    def digits(self, download=False):
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        df = pd.DataFrame(data=X)
        df['target'] = y
        if download:
            df.to_csv('digits.csv')
        return df

    def iris(self, download=False):
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        df = pd.DataFrame(data=X)
        df['target'] = y
        if download:
            df.to_csv('iris.csv')
        return df

    def wine(self, download=False):
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)
        df = pd.DataFrame(data=X)
        df['target'] = y
        if download:
            df.to_csv('wine.csv')
        return df
