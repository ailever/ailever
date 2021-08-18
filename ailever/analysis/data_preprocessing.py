import pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.storage_box = list()

    def time_splitor(self, table, only_transform=False, keep=False):
        assert 'date' in table.columns, "Table must has 'date' column"
        origin_columns = table.columns
        table = table.copy()

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
        table = table.reset_index()

        if only_transform:
            columns = table.columns.tolist()
            for origin_column in origin_columns:
                columns.pop(columns.index(origin_column))
            table = table[columns]

        if keep:
            columns = table.columns.tolist()
            for origin_column in origin_columns:
                columns.pop(columns.index(origin_column))
            self.storage_box.append(table[columns])

        return table

    def missing_value():
        pass

