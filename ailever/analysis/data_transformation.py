from .data_preprocessing import DataPreprocessor

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
        assert numeric_target_columns in table.columns, 'Target columns(numeric_target_columns) must be defined.'
        if not isinstance(numeric_target_columns, list):
            numeric_target_columns = list(numeric_target_columns)
        if not isinstance(bins, list):
            bins = list(bins)

        for target_column in numeric_target_columns:
            if not(table.dtypes[target_column] == float or table.dtypes[target_column] == int):
                table[target_column] = table[target_column].astype(float)
            for num_bin in bins:
                _, threshold = pd.cut(table[target_column], bins=num_bin, precision=6, retbins=True)
                table[target_column+f'_ew_{num_bin}bins'] = pd.cut(table[target_column], bins=num_bin, labels=threshold, precision=6, retbins=False).astype(float)
        
        return table

    @staticmethod
    def ef_binning(table, numeric_target_columns=None, bins=4):
        assert numeric_target_columns in table.columns, 'Target columns(numeric_target_columns) must be defined.'
        if not isinstance(numeric_target_columns, list):
            numeric_target_columns = list(numeric_target_columns)
        if not isinstance(bins, list):
            bins = list(bins)

        for target_column in numeric_target_columns:
            if not(table.dtypes[target_column] == float or table.dtypes[target_column] == int):
                table[target_column] = table[target_column].astype(float)
            for num_bin in bins:
                _, threshold = pd.qcut(table[target_column], q=num_bin, precision=6, retbins=True)
                table[target_column+f'_ef_{num_bin}bins'] = pd.qcut(table[target_column], q=num_bin, labels=threshold, precision=6, retbins=False).astype(float)

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
