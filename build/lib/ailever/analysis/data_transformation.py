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
    def principal_component():
        pass


class DataDiscretizor:
    # https://pbpython.com/pandas-qcut-cut.html
    @staticmethod
    def ew_binning():
        # historgam : equal width
        # percentile : equal frequency
        # v-optimal
        # diff
        return None

    @staticmethod
    def ef_binning():
        # historgam : equal width
        # percentile : equal frequency
        # v-optimal
        # diff
        return None

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



class DataTransformer(DataScaler, DataDiscretizor):
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
