
def FeatureSelection(X):
    r"""
    Example:
        >>> from ailever.forecast import FeatureSelection
        >>> fs = FeatureSelection(OneColumnData)
        >>> transformed_data = fs.univariate_feature_selection()
        >>> transformed_data = fs.recursive_feature_elimination()
        >>> transformed_data = fs.principal_component_analysis()
    """
    from .feature_selection import FeatureSelection
    return FeatureSelection(X)

def ModelSelection():
    r"""
    Example:
        >>> from ailever.forecast import ModelSelection
        >>> ms = ModelSelection(OneColumnData)
    """
    return None






def regressor(norm):
    r"""
    """
    from ._stattools import regressor
    return regressor(norm=norm)

def TSA(TS, lag=1, select_col=0, visualize=True):
    r"""
    """
    from .tsa import TSA
    return TSA(TS=TS, lag=lag, select_col=select_col, visualize=visualize)

def RSDA():
    r"""
    """
    from .ts_residual import RSDA
    return RSDA()



