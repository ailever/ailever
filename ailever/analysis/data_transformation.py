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
    def ew_binning(table, target_column=None, bins=4):
        table[target_column+f'_ew_{bins}bins'] = pd.cut(table[target_column], bins=bins, precision=6).astype(str)
        # historgam : equal width
        # percentile : equal frequency
        # v-optimal
        # diff
        return table

    @staticmethod
    def ef_binning(table, target_column=None, bins=4):
        assert target_column in table.columns, 'Target column(target_column) must be defined.'
        
        table[target_column+f'_ef_{bins}bins'] = pd.qcut(table[target_column], q=bins, precision=6).astype(str)
        # historgam : equal width
        # percentile : equal frequency
        # v-optimal
        # diff
        return table

    @staticmethod
    def opt_binning():
        # historgam : equal width
        # percentile : equal frequency
        # v-optimal
        # diff
        return None

    @staticmethod
    def diff_binning():
        # historgam : equal width
        # percentile : equal frequency
        # v-optimal
        # diff
        return None

    @staticmethod
    def padding():
        # historgam : equal width
        # percentile : equal frequency
        # v-optimal
        # diff
        return None

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
