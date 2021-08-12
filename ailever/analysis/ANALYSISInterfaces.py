from .data_processing import DataControlBlock
from .data_preprocessing import DataPreprocessor
from .data_transformation import DataTransformer
from .exploratory_data_analysis import ExploratoryDataAnalysis
from .exploratory_data_analysis import Counting

def DataProcessor(frames):
    return DataControlBlock(frames)

def EDA(frame, save=False, path='ExploratoryDataAnalysis'):
    return ExploratoryDataAnalysis(frame=frame, save=save, path=path)

def counting(frame, view='table', save=False, path=None):
    counter = Counting(frame=frame)
    frame = counter.CountUniqueValues(view=view, save=save, path=path)
    return frame
