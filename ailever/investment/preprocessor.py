from os import rename
import pandas as pd
from pandas.core.frame import DataFrame

class Preprocessor(object):
    
    dataframe = None
    base_column = None
    date_column = None
    base_period = None

    overnight = None
    rolling = None

    result = None
    

    def __init__(self, dataframe, date_column='date'):

        self.dataframe = self._lower(dataframe)
        self.date_column = date_column
        self.base_column = [x for x in self.dataframe.columns.to_list() if x != self.date_column]
        self.base_period = self.dataframe[date_column]

    def _lower(self, dataframe):
        
        df = dataframe
        df.columns = list(map(lambda x: x.lower(), df.columns))
        self.dataframe = df
        
        return self.dataframe

    def _reset(self, dataframe):
        
        self.dataframe = self._lower(dataframe)
        
        return self

    def _concat(self, **frame):
        pass

    def _rounder(self, data, option="round"):
    
        pass

    def _missing_values(self, dataframe):
        
        pass

    def _overnight(self, output="pdframe"):
        
        df = self.dataframe
        df_cross = pd.concat([df.open, df.close.shift()], axis=1)
        df_overnight = df_cross.assign(overnight=lambda x: (x['open']/x['close']) -1)['overnight']
        
        self.result = pd.concat([df, df_overnight], axis=1)
        
        if output == "pdframe":
            self.overnight = pd.concat([df[self.date_column], df_overnight], axis=1)
        if output == "ndarray":
            self.overnight = df_overnight.to_numpy()

        return self

    def _rolling(self, column="close", window=10, rolling_type="mean", output="pdframe"):

        df = self.dataframe
        df_rolling = df[column].rolling(window=window,
                                min_periods=None,
                                center=False,
                                win_type=None,
                                axis=0,
                                closed=None)
        if rolling_type =="mean":
            new_column = column + "_" + str(window) + rolling_type
            self.result = pd.concat([df, df_rolling.mean().rename(new_column, inplace=True)], axis=1)             
            
            if output =="pdframe":
                self.rolling = pd.concat([df[self.date_column], df_rolling.mean().rename(new_column, inplace=True)], axis=1)            
            if output =="ndarray":
                self.rolling = df_rolling.mean().to_numpy()

        if rolling_type =="sum":
            new_column = column + "_" + str(window) + rolling_type
            self.result = pd.concat([df, df_rolling.sum().rename(new_column, inplace=True)], axis=1)             

            if output =="pdframe":
                self.rolling = pd.concat([df[self.date_column], df_rolling.sum().rename(new_column, inplace=True)], axis=1)            
            if output =="ndarray":
                self.rolling = df_rolling.sum().to_numpy()
            
        return(self)

    def _relative(self):
        pass


    def _stochastic(self):
        pass
