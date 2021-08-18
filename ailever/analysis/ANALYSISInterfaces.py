from .data_processing import DataControlBlock
from .data_preprocessing import DataPreprocessor
from .data_transformation import DataTransformer
from .exploratory_data_analysis import ExploratoryDataAnalysis
from .exploratory_data_analysis import Counting

_dp = DataPreprocessor()
time_splitor = _dp.time_splitor
DataTransformer = DataTransformer()

def DataProcessor(frames):
    return DataControlBlock(frames)

def EDA(frame, save=False, path='ExploratoryDataAnalysis', verbose=True):
    return ExploratoryDataAnalysis(frame=frame, save=save, path=path, verbose=verbose)

def counting(frame, view='table', save=False, path=None):
    counter = Counting(frame=frame)
    frame = counter.CountUniqueValues(view=view, save=save, path=path)
    return frame
