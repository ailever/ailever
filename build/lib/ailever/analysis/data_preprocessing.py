import pandas as pd

class DataPreprocessor:
    @staticmethod
    def time_splitor(table):
        assert 'date' in table.columns, "Table must has 'date' column"
        table['date'] = pd.to_datetime(table['date'].astype(str))
        
        table = table.set_index('date')
        table['TS_year'] = table.index.year
        table['TS_quarter'] = table.index.quarter
        table['TS_month'] = table.index.month
        table['TS_daysinmonth'] = table.index.daysinmonth
        table['TS_week'] = table.index.isocalendar().week
        table['TS_weekday'] = table.index.weekday
        table['TS_day'] = table.index.day
        table['TS_hour'] = table.index.hour
        table['TS_minute'] = table.index.minute
        table['TS_second'] = table.index.second
        return table

    def missing_value():
        pass

