import numpy as np
import pandas as pd

class UCI_MLR:
    def adult(self, download=False):
        print('https://archive.ics.uci.edu/ml/datasets/Adult') 
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
        df = pd.concat([df.columns.to_frame().T, df], axis=0).reset_index().drop('index', axis=1)
        df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '50K']

        if download:
            df.to_csv('adult.csv')
        return df

    def beijing_airquality(self, download=False):
        print('https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data')
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv')
        
        if download:
            df.to_csv('beijing_airquality.csv')
        return df
        
    def white_wine(self, download=False):
        print('https://archive.ics.uci.edu/ml/datasets/Wine') 
        df_white = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')

        white_columns = list()
        for white_column in df_white.columns[0].split(';'):
            if r'"' in list(white_column):
                white_columns.append(white_column[1:-1])
            else:
                white_columns.append(white_column)
            
        white_rows = list()
        for i in range(df_white.shape[0]):
            white_rows.append(df_white.loc[i, df_white.columns[0]].split(';'))

        df = pd.DataFrame(data=white_rows, columns=white_columns)
        
        if download:
            df.to_csv('white_wine.csv')
        return df

    def red_wine(self, download=False):
        print('https://archive.ics.uci.edu/ml/datasets/Wine') 
        df_red = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv')

        red_columns = list()
        for red_column in df_red.columns[0].split(';'):
            if r'"' in list(red_column):
                red_columns.append(red_column[1:-1])
            else:
                red_columns.append(red_column)
            
        red_rows = list()
        for i in range(df_red.shape[0]):
            red_rows.append(df_red.loc[i, df_red.columns[0]].split(';'))

        df = pd.DataFrame(data=red_rows, columns=red_columns)

        if download:
            df.to_csv('red_wine.csv')
        return df

    def annealing(self, download=False):
        print('https://archive.ics.uci.edu/ml/datasets/Annealing') 
        csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data')
        first_row = csv.columns.values[np.newaxis, :]

        columns=['family', 'product-type', 'steel', 'carbon', 'hardness', 'temper_rolling', 'condition', 'formability', 'strength', 'non-ageing', 'surface-finish', 'surface-quality', 'enamelability', 'bc', 'df', 'bt', 'bw/me', 'bl', 'm', 'chrom', 'phos', 'cbond', 'marvi', 'exptl', 'ferro', 'corr', 'blue/bright/varn/clean', 'lustre', 'jurofm','s','p','shape','thick', 'width', 'len', 'oil', 'bore', 'packing', 'classes']
        csv.columns = columns
        df = pd.DataFrame(data=first_row, columns=columns)
        df = df.append(csv)
        
        if download:
            df.to_csv('annealing.csv')
        return df


    def breast_cancer(self, download=False):
        print('https://archive.ics.uci.edu/ml/datasets/Breast+Cancer')
        breast = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data')
        columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
        base = pd.DataFrame(data=breast.iloc[0], columns=columns)
        breast.columns = columns
        df = base.append(breast)

        if download:
            df.to_csv('breast_cancer.csv')
        return df
