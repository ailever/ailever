from .digitization_for_categorical_variables import CategoricalDataset, QuantifyingModel, Criterion

import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt


class DataPreprocessor:
    def __init__(self):
        self.storage_box = list()

    def time_splitor(self, table, date_column=None, only_transform=False, keep=False):
        origin_columns = table.columns
        table = table.copy()

        if date_column is None:
            assert 'date' in table.columns, "Table must has 'date' column"
            table['date'] = pd.to_datetime(table['date'].astype(str))
            table = table.set_index('date')
        else:
            table[date_column] = pd.to_datetime(table[date_column].astype(str))
            table = table.set_index(date_column)

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
        table['TS_sequence'] = np.linspace(-1, 1, table.shape[0])
        table = table.reset_index()
        
        dropping_columns = list()
        for column in table.columns:
            num_unique = pd.unique(table[column]).shape[0]
            if num_unique == 1:
                dropping_columns.append(column)
        for d_column in dropping_columns:
            table = table.drop(d_column, axis=1)
        
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

    def sequence_smoothing(self, table, target_column=None, date_column=None, freq='D', smoothing_order=1, decimal=None, including_model_object=False, only_transform=False, keep=False):
        assert target_column is not None, 'Target column must be defined. Set a target(target_column) on columns of your table'

        origin_columns = table.columns
        table = table.copy()

        if date_column is None:
            assert 'date' in table.columns, "Table must has 'date' column"
            table['date'] = pd.to_datetime(table['date'].astype(str))
            table = table.set_index('date')
        else:
            table[date_column] = pd.to_datetime(table[date_column].astype(str))
            table = table.set_index(date_column)

        table = table.asfreq(freq).fillna(method='bfill').fillna(method='ffill')
        
        trend_orders = [(1,0,0), (0,0,1), (1,0,1),
                        (2,0,0), (2,0,1), (0,0,2), (1,0,2), (2,0,2),
                        (0,1,0), (1,1,0), (0,1,1), (1,1,1),
                        (2,1,0), (2,1,1), (0,1,2), (1,1,2), (2,1,2), 
                        (0,2,0), (1,2,0), (0,2,1), (1,2,1), 
                        (2,2,0), (2,2,1), (0,2,2), (1,2,2), (2,2,2)]
        trend_orders = list(filter(lambda x: x[1] <= smoothing_order, trend_orders))
        if freq in ['D']:
            seasonal_orders = [(0,0,0,0), (0,1,0,7)]
        elif freq in ['B']:
            seasonal_orders = [(0,0,0,0), (0,1,0,5)]
        elif freq in ['W']:
            seasonal_orders = [(0,0,0,0), (0,1,0,4)]
        elif freq in ['M', 'MS', 'BM', 'BMS']:
            seasonal_orders = [(0,0,0,0), (0,1,0,12)]
        elif freq in ['Q', 'QS', 'BQ', 'BQS']:
            seasonal_orders = [(0,0,0,0), (0,1,0,4)]
        elif freq in ['A', 'AS']:
            seasonal_orders = [(0,0,0,0), (0,1,0,10)]
        else:
            seasonal_orders = [(0,0,0,0)]
        seasonal_orders = list(filter(lambda x: x[1] <= smoothing_order, seasonal_orders))

        smoothing_models = dict()
        freq = table.index.freq
        for seasonal_order in seasonal_orders:
            for trend_order in trend_orders:
                model = smt.SARIMAX(table[target_column], order=trend_order, seasonal_order=seasonal_order, trend=None, freq=freq, simple_differencing=False)
                model = model.fit(disp=False)
                column_name = target_column + '_smt' + str(trend_order).strip('()').replace(', ', '') + 'X' + str(seasonal_order).strip('()').replace(', ', '')
                if model.mle_retvals['converged']:
                    print(f'* {trend_order}X{seasonal_order} : CONVERGENT')
                    if decimal:
                        table[column_name] = model.predict().round(decimal)
                    else:
                        table[column_name] = model.predict()

                else:
                    print(f'* {trend_order}X{seasonal_order} : DIVERGENT')

                if including_model_object:
                    smoothing_models[column_name] = model

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

        if including_model_object:
            return table, smoothing_models
        else:
            return table

    def missing_value(self):
        pass

    def to_numeric(self, table, target_column=None, only_transform=False, keep=False):
        assert target_column is not None, 'Target column must be defined. Set a target(target_column) on columns of your table'

        origin_columns = table.columns
        table = table.copy()

        training_information = dict()
        training_information['NumUnique'] = df['education'].unique().shape[0]
        training_information['NumFeature'] = 3
        training_information['Epochs'] = 1000

        dataset = CategoricalDataset(X=df['education'])
        model = Model(training_information)
        optimizer = optim.Adamax(model.parameters(), lr=0.01)
        criterion = Criterion()

        epochs = training_information['Epochs']
        for epoch in range(epochs):
            losses = list()
            for train, target in dataset:
                hypothesis = model(train)
                cost = criterion(hypothesis, target)
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                losses.append(cost.data.item())
            if epoch% 100 == 0:
                print(sum(losses))
        
        feature_columns = list(map(lambda x: f'{target_column}_f'+str(x), list(range(training_information['NumFeature']))))
        feature_frame = pd.DataFrame(data=model.latent_feature.numpy(), columns=feature_columns)
        table = pd.concat([table, feature_frame], axis=1)       

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
