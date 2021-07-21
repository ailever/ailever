from .exploratory_data_analysis import ExploratoryDataAnalysis
from .exploratory_data_analysis import Counting

def EDA(frame, save=False, path='ExploratoryDataAnalysis'):
    return ExploratoryDataAnalysis(frame=frame)

def counting(frame, view='table', save=False, path=None):
    counter = Counting(frame=frame)
    frame = counter.CountsUniqueValues(view=view, save=save, path=path)
    return frame
