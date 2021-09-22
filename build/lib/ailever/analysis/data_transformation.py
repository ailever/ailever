from .data_preprocessing import DataPreprocessor

import numpy as np
import pandas as pd


class DataReduction:
    def __init__(self):
        self.storage_box = list()

    @staticmethod
    def recursive_partitioning():
        pass

    @staticmethod
    def pruning():
        pass

    @staticmethod
    def dimenstion_reduction():
        pass

    @staticmethod
    def discrete_wavelet_transform():
        pass

    @staticmethod
    def PCA():
        pass

    @staticmethod
    def SVD():
        pass

    @staticmethod
    def LDA():
        pass

class DataDiscretizor:
    def __init__(self):
        self.storage_box = list()

    # https://pbpython.com/pandas-qcut-cut.html
    def ew_binning(self, table, target_columns=None, bins=4, replace=False, only_transform=False, keep=False):
        numeric_target_columns = target_columns
        origin_columns = table.columns
        table = table.copy()

        if not isinstance(numeric_target_columns, list):
            numeric_target_columns = [numeric_target_columns]
        if not isinstance(bins, list):
            bins = [bins]
        for numeric_target_column in numeric_target_columns:
            assert numeric_target_column in table.columns, 'Each target column(through numeric target_columns) must be correctly defined.'


        for target_column in numeric_target_columns:
            if not(table.dtypes[target_column] == float or table.dtypes[target_column] == int):
                table[target_column] = table[target_column].astype(float)
            for num_bin in bins:
                _, threshold = pd.cut(table[target_column], bins=num_bin, precision=6, retbins=True)
                table[target_column+f'_ew{num_bin}bins'] = pd.cut(table[target_column], bins=num_bin, labels=threshold[1:], precision=6, retbins=False).astype(float)
        
        if replace:
            print('If you want to get only the transform result, set replace=False. And the replace option is only valid for first thing in transformed columns with first bins.')
            columns = table.columns.tolist()
            for origin_column in origin_columns:
                columns.pop(columns.index(origin_column))
            table_ = table[origin_columns].copy()
            table_[target_columns[0]] = table[columns[0]]
            table = table_

        elif only_transform:
            columns = table.columns.tolist()
            for origin_column in origin_columns:
                columns.pop(columns.index(origin_column))
            table = table[columns]

        if keep:
            columns = table.columns.tolist()
            for origin_column in origin_columns:
                columns.pop(columns.index(origin_column))
            self.storage_box.append(table[columns])
        return table

    def ef_binning(self, table, target_columns=None, bins=4, replace=False, only_transform=False, keep=False):
        numeric_target_columns = target_columns
        origin_columns = table.columns
        table = table.copy()

        if not isinstance(numeric_target_columns, list):
            numeric_target_columns = [numeric_target_columns]
        if not isinstance(bins, list):
            bins = [bins]
        for numeric_target_column in numeric_target_columns:
            assert numeric_target_column in table.columns, 'Each target columns(numeric target_columns) must be correctly defined.'
        

        def retbins_transform(bins):
            bins_series = pd.Series(bins).astype(str)
            duplication = pd.DataFrame(bins_series.value_counts(), columns=['count'])
            duplication['transform'] = duplication['count'].apply(lambda x: [ i+1 for i in range(x)])

            bins_frame = pd.DataFrame(bins_series, columns=['bins'])
            bins_frame['transform'] = bins_frame['bins'].apply(lambda x: duplication['transform'][x])
            base = 0
            memory = None
            for idx, row in bins_frame.iterrows():
                if idx != bins_frame.index[0]:
                    if memory != row['bins']:
                        base = 0
                if len(row['transform']) != 1:
                    bins_frame.at[idx, 'transform'] = row['bins']+'_'+str(row['transform'][base])
                    base += 1
                else:
                    bins_frame.at[idx, 'transform'] = row['bins']
                memory = row['bins']
            return bins_frame.transform.values

        for target_column in numeric_target_columns:
            if not(table.dtypes[target_column] == float or table.dtypes[target_column] == int):
                table[target_column] = table[target_column].astype(float)
            for num_bin in bins:
                _, threshold = pd.qcut(table[target_column], q=num_bin, precision=6, duplicates='drop', retbins=True)
                table[target_column+f'_ef{num_bin}bins'] = pd.qcut(table[target_column], q=num_bin, labels=threshold[1:], precision=6, duplicates='drop', retbins=False).astype(float)
                if threshold.shape[0] != num_bin + 1:
                    print(f'Some bins of target column {target_column} are duplicated during binning with {num_bin}.')

        if replace:
            print('If you want to get only the transform result, set replace=False. And the replace option is only valid for first thing in transformed columns with first bins.')
            columns = table.columns.tolist()
            for origin_column in origin_columns:
                columns.pop(columns.index(origin_column))
            table_ = table[origin_columns].copy()
            table_[target_columns[0]] = table[columns[0]]
            table = table_

        elif only_transform:
            columns = table.columns.tolist()
            for origin_column in origin_columns:
                columns.pop(columns.index(origin_column))
            table = table[columns]

        if keep:
            columns = table.columns.tolist()
            for origin_column in origin_columns:
                columns.pop(columns.index(origin_column))
            self.storage_box.append(table[columns])

        return table

    @staticmethod
    def opt_binning(table, target_columns=None):
        # historgam : equal width
        # percentile : equal frequency
        # v-optimal
        # diff
        return table

    @staticmethod
    def diff_dw_binning(table, target_columns=None):
        # historgam : equal width
        # percentile : equal frequency
        # v-optimal
        # diff
        return table

    @staticmethod
    def diff_df_binning(table, target_columns=None):
        # historgam : equal width
        # percentile : equal frequency
        # v-optimal
        # diff
        return table

    def padding(self, table, target_column=None, target_instance=None, non_target_pad=None):
        origin_columns = table.columns
        table = table.copy()
        assert target_column, 'Target column(target_column) must be defined.'
        assert target_instance, 'Target instance(target_instance) must be defined.'
        if non_target_pad is None:
            table = table[target_column].apply(lambda x: x if x==target_instance else 0)
        else:
            table = table[target_column].apply(lambda x: x if x==target_instance else non_target_pad)

        return table

    def derivatives(self, table, target_columns=None, only_transform=False, keep=False):
        numeric_target_columns = target_columns
        origin_columns = table.columns
        table = table.copy()
        if not isinstance(numeric_target_columns, list):
            numeric_target_columns = [numeric_target_columns]
        for numeric_target_column in numeric_target_columns:
            assert numeric_target_column in table.columns, 'Each target columns(numeric_target_columns) must be correctly defined.'
            table[numeric_target_column] = table[numeric_target_column].astype(float)
            table[numeric_target_column+'_increasing_1st'] = table[numeric_target_column].diff().fillna(0).apply(lambda x: 1 if x>0 else 0)
            table[numeric_target_column+'_increasing_2nd'] = table[numeric_target_column+'_increasing_1st'].diff().fillna(0).apply(lambda x: 1 if x>0 else 0)

        if only_transform:
            columns = table.columns.tolist()
            for origin_column in origin_columns:
                columns.pop(columns.index(origin_column))
            table = table[columns]

        if keep:
            columns = table.columns.tolist()
            for origin_column in origin_columns:
                columns.pop(columns.index(origin_column))
            self.storage_box.append(table[columns])

        return table


    @staticmethod
    def entropy():
        pass

    @staticmethod
    def chisqure():
        pass

    @staticmethod
    def clustering():
        pass


class DataScaler:
    def __init__(self):
        self.storage_box = list()

    @staticmethod
    def minmax_normalization():
        pass

    @staticmethod
    def zscore_normalization():
        pass

    @staticmethod
    def robust_normalization():
        pass

    @staticmethod
    def decimal_scaling():
        pass



class DataTransformer(DataScaler, DataDiscretizor, DataPreprocessor):
    def __init__(self):
        self.storage_box = list()

    @staticmethod
    def regression():
        pass

    @staticmethod
    def aggregation():
        pass

    @staticmethod
    def binning():
        pass

    @staticmethod
    def conceptual():
        pass

    def build(self):
        table = pd.concat(self.storage_box, axis=1)
        return table
    
    def empty(self):
        self.storage_box = list()


