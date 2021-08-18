from .data_preprocessing import DataPreprocessor

import numpy as np
import pandas as pd


class DataReduction:
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
    # https://pbpython.com/pandas-qcut-cut.html
    @staticmethod
    def ew_binning(table, numeric_target_columns=None, bins=4):
        if not isinstance(numeric_target_columns, list):
            numeric_target_columns = [numeric_target_columns]
        if not isinstance(bins, list):
            bins = [bins]
        for target_column in numeric_target_columns:
            assert target_column in table.columns, 'Each target column(through numeric_target_columns) must be correctly defined.'

        for target_column in numeric_target_columns:
            if not(table.dtypes[target_column] == float or table.dtypes[target_column] == int):
                table[target_column] = table[target_column].astype(float)
            for num_bin in bins:
                _, threshold = pd.cut(table[target_column], bins=num_bin, precision=6, retbins=True)
                table[target_column+f'_ew_{num_bin}bins'] = pd.cut(table[target_column], bins=num_bin, labels=threshold[1:], precision=6, retbins=False).astype(float)
        
        return table

    @staticmethod
    def ef_binning(table, numeric_target_columns=None, bins=4):
        if not isinstance(numeric_target_columns, list):
            numeric_target_columns = [numeric_target_columns]
        if not isinstance(bins, list):
            bins = [bins]
        for target_column in numeric_target_columns:
            assert target_column in table.columns, 'Each target columns(numeric_target_columns) must be correctly defined.'
        
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
                _, threshold = pd.qcut(table[target_column], q=num_bin, precision=6, retbins=True)
                if threshold.shape[0] == np.unique(threshold):
                    table[target_column+f'_ef_{num_bin}bins'] = pd.qcut(table[target_column], q=num_bin, labels=threshold[1:], precision=6, retbins=False).astype(float)
                else:
                    threshold = retbins_transform(threshold)
                    table[target_column+f'_ef_{num_bin}bins'] = pd.qcut(table[target_column], q=num_bin, labels=threshold[1:], precision=6, retbins=False).astype(str)

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

    @staticmethod
    def padding(table, target_column=None, target_instance=None, non_target_pad=None):
        assert target_column, 'Target column(target_column) must be defined.'
        assert target_instance, 'Target instance(target_instance) must be defined.'

        if non_target_pad is None:
            table = table[target_column].apply(lambda x: x if x==target_instance else 0)
        else:
            table = table[target_column].apply(lambda x: x if x==target_instance else non_target_pad)
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
