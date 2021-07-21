import os
from .exploratory_data_analysis import ExploratoryDataAnalysis
from .exploratory_data_analysis import Counting

def EDA(frame, path='ExploratoryDataAnalysis', save=False):
    if not os.path.isdir(path):
        os.mkdir(path)
    
    return ExploratoryDataAnalysis(frame)


def counting(frame, path=None, save=False):
    frame = Counting(frame)
    frame = frame.CountsByInstance(path=path, save=save)
    return frame
