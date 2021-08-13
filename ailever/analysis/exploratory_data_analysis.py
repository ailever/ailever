import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-whitegrid')

class ExploratoryDataAnalysis:
    def __init__(self, frame, save=False, path='ExploratoryDataAnalysis', type_info=True, verbose:bool=True):
        self.frame = frame
        self.path = path
        
        data = np.array([["eda.table_definition()", ""],
                         ["eda.cleaning()", ""],
                         ["eda.attributes_specification()", ""],
                         ["eda.univariate_frequency()", ""],
                         ["eda.univariate_percentile()", ""],
                         ["eda.univariate_conditional_frequency()", ""],
                         ["eda.univariate_conditional_percentile()", ""],
                         ["eda.multivariate_frequency()", ""],
                         ["eda.information_value()", ""],
                         ["eda.feature_importance()", ""]])
        if verbose:
            print('* Column Date Types')
            print(frame.dtypes)
            print('\n* EDA object method list')
            print(pd.DataFrame(data=data, columns=['Commands', 'Description']).set_index('Commands'))

        if save:
            self._excel()     

    def cleaning(self, priority_frame=None, save=False, path=None, saving_name=None, as_float:list=None, as_int:list=None, as_category:list=None, as_str:list=None, as_date:list=None, verbose:bool=False):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame

        """ Core """
        cleaning_failures = list()
        # base clearning
        for column in table.columns:
            if table[column].dtype == 'object':
                table[column] = table[column].astype(str)
            elif table[column].dtype == 'float':
                table[column] = table[column].astype(float)
            elif table[column].dtype == 'int':
                table[column] = table[column].astype(int)
            elif table[column].dtype == 'category':
                table[column] = table[column].astype('category')


        # all type-cleaning
        if as_int is all:
            for column in table.columns:
                try:
                    table[column] = table[column].astype(int)
                except:
                    pass
            as_int = None
        if as_float is all:
            for column in table.columns:
                try:
                    table[column] = table[column].astype(float)
                except:
                    pass
            as_float = None
        if as_category is all:
            for column in table.columns:
                table[column] = table[column].astype('category')
            as_category = None
        if as_str is all:
            for column in table.columns:
                table[column] = table[column].astype(str)
            as_str = None

        converting_failures = list()
        # to convert as float data-type
        if as_float is not None:
            if isinstance(as_float, str):
                as_float = list(as_float)
            for column in as_float:
                try:
                    table[column] = table[column].astype(float)
                except:
                    converting_failures.append(column)
        # to convert as int data-type
        if as_int is not None:
            if isinstance(as_int, str):
                as_int = list(as_int)
            for column in as_int:
                try:
                    table[column] = table[column].astype(int)
                except:
                    converting_failures.append(column)
        # to convert as category data-type
        if as_category is not None:
            if isinstance(as_category, str):
                as_category = list(as_category)
            for column in as_category:
                try:
                    table[column] = table[column].astype('category')
                except:
                    converting_failures.append(column)
        # to convert as str data-type
        if as_str is not None:
            if isinstance(as_str, str):
                as_str = list(as_str)
            for column in as_str:
                try:
                    table[column] = table[column].astype(str)
                except:
                    converting_failures.append(column)
        # to convert as datetime64 data-type
        if as_date is not None:
            if isinstance(as_date, str):
                as_date = list(as_date)
            for column in as_date:
                try:
                    table[column] = pd.to_datetime(table[column].astype(str))
                except:
                    converting_failures.append(column)
        
        if cleaning_failures:
            print(f'Cleaning failure list about changing data-type: {cleaning_failures}')
        if converting_failures:
            print(f'Converting failure list about changing data-type: {converting_failures}')
        """ Core """

        if priority_frame is None:
            self.frame = table

        if verbose:
            return self.attributes_specification(priority_frame=priority_frame, save=False, path=None, saving_name=None, view='-')
        return table

    
    def _excel(self, priority_frame=None, save=False, path=None, saving_name=None):
        pass


    def table_definition(self, priority_frame=None, save=False, path=None, saving_name=None, view='summary'):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame

        """ Core """
        base_columns = ['NumRows', 'NumColumns', 'NumNumericColumnType', 'NumCategoricalColumnType', ]
        lettertype_columns = table.columns[(table.dtypes == 'category') | (table.dtypes == 'object')]
        table_definition = pd.DataFrame(columns=base_columns)
        C = 0; N = 0
        for column in table.columns:
            ColumnType = 'Letter' if column in lettertype_columns else 'Number' 
            if ColumnType == 'Letter':
                C += 1
            else:
                N += 1
        a_row = pd.DataFrame(data=[[table.shape[0], table.shape[1], N, C]], columns=base_columns)
        table_definition = table_definition.append(a_row)
        """ Core """

        saving_name = f'{saving_name}_EDA_TableDefinition.csv' if saving_name is not None else 'EDA_TableDefinition.csv'
        _csv_saving(table_definition, save, self.path, path, saving_name)
        return table_definition


    def attributes_specification(self, priority_frame=None, save=False, path=None, saving_name=None, visual='on'):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame

        """ Core """
        base_columns = ['Column', 'ColumnType', 'NumUniqueInstance', 'NumMV', 'DataType', 'DataExample', 'MaxInstanceLength']
        lettertype_columns = table.columns[(table.dtypes == 'category') | (table.dtypes == 'object')]
        attributes_matrix = pd.DataFrame(columns=base_columns)
        for column in table.columns:
            ColumnType = 'Letter' if column in lettertype_columns else 'Number' 
            NumUniqueInstance = table[column].value_counts().shape[0]
            MaxInstanceLength = table[column].astype('str').apply(lambda x: len(x)).max()
            NumMV = table[column].isna().sum()
            DataType = table[column].dtype.type
            a_row = pd.DataFrame(data=[[column, ColumnType, NumUniqueInstance, NumMV, DataType, table[column].iloc[np.random.randint(NumUniqueInstance)], MaxInstanceLength]], columns=base_columns)
            attributes_matrix = attributes_matrix.append(a_row)

        attributes_matrix.insert(2, 'NumRows', table.shape[0])
        attributes_matrix.insert(4, 'IdealSymmericCount', table.shape[0]/attributes_matrix['NumUniqueInstance'])
        attributes_matrix.insert(5, 'IdealSymmericRatio', 1/attributes_matrix['NumUniqueInstance'])    
        attributes_matrix.insert(6, 'MVRate', attributes_matrix['NumMV']/table.shape[0])
        attributes_matrix = attributes_matrix.reset_index().drop('index', axis=1)
        """ Core """

        saving_name = f'{saving_name}_EDA_AttributesSpecification.csv' if saving_name is not None else 'EDA_AttributesSpecification.csv'
        _csv_saving(attributes_matrix, save, self.path, path, saving_name)

        # Visualization
        if visual == 'on':
            gridcols = 3
            table_columns = self.frame.columns
            num_columns = (table_columns.shape[0])
            quotient = num_columns//gridcols
            reminder = num_columns%gridcols
            with plt.style.context('seaborn-whitegrid'):
                layout = (quotient, gridcols) if reminder==0 else (quotient+1, gridcols)
                plt.figure(figsize=(25, layout[0]*5))
                axes = dict()
                for i in range(0, layout[0]):
                    for j in range(0, layout[1]):
                        idx = i*layout[1] + j
                        axes[idx]= plt.subplot2grid(layout, (i, j))
                for idx, column in enumerate(table_columns):
                    table[column].hist(ax=axes[idx], xrot=30, edgecolor='black')
                    axes[idx].set_title(column)
                plt.tight_layout()
        return attributes_matrix


    def descriptive_statistics(self, priority_frame=None, save=False, path=None, saving_name=None):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame
        
        describing_matrix = table.describe().T

        saving_name = f'{saving_name}_EDA_DescriptiveStatistics.csv' if saving_name is not None else 'EDA_DescriptiveStatistics.csv'
        _csv_saving(describing_matrix, save, self.path, path, saving_name)

        return describing_matrix

    def univariate_frequency(self, priority_frame=None, save=False, path=None, saving_name=None, mode='base', view='summary'):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame
        

        """ Core """
        category_columns = table.columns[table.dtypes == 'category']
        for column in category_columns:
            table[column] = table[column].astype(str)

        base_columns = ['Column', 'NumMV', 'Instance', 'NumUniqueInstance', 'Count', 'Rank']
        frequency_matrix = pd.DataFrame(columns=base_columns)
        for column in table.columns:
            instance_count = table.value_counts(column, ascending=False).to_frame()
            rank_mapper = instance_count.rank(ascending=False)
            instance_frequency = instance_count.sort_index(ascending=True).reset_index().rename(columns={column:'Instance', 0:'Count'})
            instance_frequency.insert(0, 'Column', column)
            instance_frequency.insert(2, 'NumMV', table[column].isna().sum())
            instance_frequency.insert(3, 'NumUniqueInstance', instance_count.shape[0] + int(bool(table[column].isna().sum())) if mode == 'missing' else instance_count.shape[0])
            instance_frequency.insert(5, 'Rank', instance_frequency.Instance.apply(lambda x: rank_mapper.loc[x]))
            frequency_matrix = frequency_matrix.append(instance_frequency)
        NumRows = table.shape[0]
        frequency_matrix.insert(1, 'NumRows', NumRows)
        frequency_matrix.insert(3, 'MVRate', frequency_matrix.NumMV/frequency_matrix.NumRows)
        frequency_matrix.insert(4, 'NumRows_EFMV', frequency_matrix.NumRows - frequency_matrix.NumMV)
        frequency_matrix.insert(7, 'IdealSymmericCount', frequency_matrix.NumRows/frequency_matrix.NumUniqueInstance)
        frequency_matrix.insert(9, 'IdealSymmericRatio', 1/frequency_matrix.NumUniqueInstance)
        frequency_matrix.insert(10, 'Ratio', frequency_matrix.Count/frequency_matrix.NumRows)

        for column in category_columns:
            table[column] = table[column].astype('category')
        """ Core """

        saving_name = f'{saving_name}_EDA_UnivariateFrequencyAnalysis.csv' if saving_name is not None else 'EDA_UnivariateFrequencyAnalysis.csv'
        _csv_saving(frequency_matrix, save, self.path, path, saving_name)

        if view == 'full':
            frequency_matrix = frequency_matrix
        elif view=='summary':
            frequency_matrix = frequency_matrix[['Column', 'Instance', 'Count', 'Rank']].sort_values(['Column', 'Rank'])
        return frequency_matrix


    def univariate_conditional_frequency(self, priority_frame=None, save=False, path=None, saving_name=None, base_column=None, view='summary'):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame

        """ Core """
        category_columns = table.columns[table.dtypes == 'category']
        for column in category_columns:
            table[column] = table[column].astype(str)

        if base_column is not None:
            base = table.groupby([base_column])[base_column].count().to_frame()
            base.columns = pd.Index(map(lambda x : (x, 'Instance') ,base.columns))
            for idx, column in enumerate(table.columns):
                if column != base_column:
                    concatenation_frame = table.groupby([base_column, column])[base_column].count().to_frame().unstack(column)
                    multi_index_frame = concatenation_frame.columns.to_frame()
                    multi_index_frame[0] = concatenation_frame.columns.names[1]
                    concatenation_frame.columns = multi_index_frame.set_index([0, column]).index                
                    base = pd.concat([base, concatenation_frame], axis=1).fillna(0)
        else:
            for idx, column in enumerate(table.columns):
                if idx == 0:
                    base_column = column
                    base = table.groupby([base_column])[base_column].count().to_frame()
                    base.columns = pd.Index(map(lambda x : (x, 'Instance') ,base.columns))
                else:
                    concatenation_frame = table.groupby([base_column, column])[base_column].count().to_frame().unstack(column)
                    multi_index_frame = concatenation_frame.columns.to_frame()
                    multi_index_frame[0] = concatenation_frame.columns.names[1]
                    concatenation_frame.columns = multi_index_frame.set_index([0, column]).index                
                    base = pd.concat([base, concatenation_frame], axis=1).fillna(0)

        base = base.reset_index()            
        base.insert(0, 'Column', base_column)
        
        columnidx = base.columns.to_frame()
        columnidxframe = pd.DataFrame(columnidx.values)
        columnidxframe.iat[0,0] = 'BaseColumn'    
        columnidxframe.iat[1,0] = 'Column'
        columnidxframe.iat[1,1] = 'Instance'
        columnidxframe.iat[2,1] = 'InstanceCount'
        base.columns = columnidxframe.set_index([0,1]).index

        for column in category_columns:
            table[column] = table[column].astype('category')
        """ Core """

        saving_name = f'{saving_name}_EDA_UnivariateConditionalFrequencyAnalysis.csv' if saving_name is not None else 'EDA_UnivariateConditionalFrequencyAnalysis.csv'
        _csv_saving(base, save, self.path, path, saving_name)
        
        print(f'[AILEVER] base_column : {base_column}')
        print(f'[AILEVER] Column list : {table.columns.to_list()}')
        return base


    def univariate_percentile(self, priority_frame=None, save=False, path=None, saving_name=None, mode='base', view='summary', percent=5, visual='on'):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame
        
        """ Core """
        # for Numeric Columns
        table = table[table.columns[(table.dtypes != 'object') & (table.dtypes != 'category')]]
        assert table.shape[1] >= 1, "This table doesn't even have a single numerical column. Change data-type of columns on table"
        
        percentile_range = list()
        percent_ = (percent/100); i = 0
        while True:
            i += 1
            cumulative_percent = round(percent_*i, 4)
            if cumulative_percent >= 1 :
                break
            percentile_range.append(cumulative_percent)
        
        self.percentiles = percentile_range
        describing_matrix = table.describe(percentiles=percentile_range).T
        describing_matrix.insert(3, 'DiffMaxMin', describing_matrix['max'] - describing_matrix['min'])
        describing_matrix.insert(4, 'Density', describing_matrix['count']/(describing_matrix['max'] - describing_matrix['min']))
        describing_matrix.insert(5, 'DiffMaxMin_PRU', (describing_matrix['max'] - describing_matrix['min'])/(len(percentile_range)+1))    
        percentile_columns = describing_matrix.columns[3:]
        
        base_columns = ['Column', 'NumRows', 'NumRows_EFMV', 'MVRate' , 'NumUniqueInstance', 'IdealSymmetricCount', 'IdealSymmetricRatio']
        data = dict()
        data['Column'] = describing_matrix.index.to_list()
        data['NumRows'] = [table.shape[0]]*describing_matrix.shape[0]
        data['NumRows_EFMV'] = describing_matrix['count'].to_list()
        data['MVRate'] = ((table.shape[0] - describing_matrix['count'])/table.shape[0]).to_list()
        data['NumUniqueInstance'] = [table.value_counts(column, ascending=False).to_frame().shape[0] + int(bool(table[column].isna().sum())) if mode == 'missing' else table.value_counts(column, ascending=False).to_frame().shape[0] for column in describing_matrix.index ]
        data['IdealSymmetricCount'] = list(map(lambda x: table.shape[0]/x, data['NumUniqueInstance'])) 
        data['IdealSymmetricRatio'] = list(map(lambda x: 1/x, data['NumUniqueInstance']))

        percentile_base = pd.DataFrame(data=data, columns=base_columns)#percentile_matrix = pd.DataFrame(columns=describing_matrix.columns[4:-1])
        percentile_matrix = pd.concat([percentile_base, describing_matrix.reset_index().drop(['index', 'count'], axis=1)], axis=1)
        
        base_column_for_diff = 'min'
        percentile_matrix_for_diff = percentile_matrix.loc[:, 'min':'max']
        for column in percentile_matrix_for_diff.columns:
            percentile_matrix[base_column_for_diff+'-'+column] = percentile_matrix_for_diff[column] - percentile_matrix_for_diff[base_column_for_diff]
            base_column_for_diff = column 
        percentile_matrix.drop('min-min', axis=1, inplace=True)
        percentile_base_matrix = percentile_matrix.loc[:, f'{percent}%':'max']
        percentile_diff_matrix = percentile_matrix.loc[:, f'min-{percent}%':]
        percentile_matrix['HighDensityRange'] = percentile_diff_matrix.columns[percentile_diff_matrix.values.argmin(axis=1)]
        percentile_matrix['HighDensityInstance'] = list(map(lambda x: percentile_base_matrix.iloc[x[0], x[1]], zip(percentile_base_matrix.index, percentile_diff_matrix.values.argmin(axis=1))))
        percentile_matrix['HighDensityMinMaxRangeRatio'] = (percentile_matrix['HighDensityInstance'] - percentile_matrix['min'])/(percentile_matrix['max'] - percentile_matrix['min'])
        """ Core """

        saving_name = f'{saving_name}_EDA_UnivariatePercentileAnalysis.csv' if saving_name is not None else 'EDA_UnivariatePercentileAnalysis.csv'
        _csv_saving(percentile_matrix, save, self.path, path, saving_name)
        

        if mode == 'base':
            percentile_matrix = percentile_matrix[percentile_matrix.columns.drop('NumRows')]
        elif mode == 'missing':
            percentile_matrix = percentile_matrix[percentile_matrix.columns.drop('NumRows_EFMV')]
        
        if visual == 'on':
            _, ax = plt.subplots(1, 1, figsize=(25, 5))
            minmax_percentile_matrix = percentile_matrix.copy().set_index('Column').loc[:, 'min':'max']
            array = minmax_percentile_matrix.values
            minmax_percentile_matrix.loc[:,'min':'max'] = (array - array[:,0][:, np.newaxis])/(array[:,-1][:, np.newaxis] - array[:,0][:, np.newaxis])
            sns.heatmap(minmax_percentile_matrix)

        if view == 'full':
            percentile_matrix = percentile_matrix
        elif view == 'p': # percentils
            percentile_matrix = pd.concat([percentile_matrix['Column'], percentile_matrix.loc[:, 'min':'max'], percentile_matrix.loc[:, 'HighDensityRange' : 'HighDensityMinMaxRangeRatio']], axis=1)
        elif view == 'ap':
            percentile_matrix = pd.concat([percentile_matrix['Column'], percentile_matrix.loc[:, 'DiffMaxMin':'max'], percentile_matrix.loc[:, 'HighDensityRange': 'HighDensityMinMaxRangeRatio']], axis=1)
        elif view == 'dp':
            percentile_matrix = pd.concat([percentile_matrix['Column'], percentile_matrix.loc[:, f'min-{percent}%':]], axis=1)
        elif view == 'adp':
            percentile_matrix = pd.concat([percentile_matrix['Column'], percentile_matrix.loc[:,'DiffMaxMin':'min'], percentile_matrix.loc[:,'max'], percentile_matrix.loc[:, f'min-{percent}%':]], axis=1)
        elif view == 'result':
            percentile_matrix = pd.concat([percentile_matrix['Column'], percentile_matrix.loc[:, 'HighDensityRange': 'HighDensityMinMaxRangeRatio']], axis=1)
        elif view == 'summary':
            percentile_matrix = percentile_matrix[['Column', 'Density', 'HighDensityRange', 'HighDensityInstance', 'HighDensityMinMaxRangeRatio', 'min', 'max', 'mean', 'std']+[ 'NumRows' if mode=='missing' else 'NumRows_EFMV']]
        return percentile_matrix


    def univariate_conditional_percentile(self, priority_frame=None, save=False, path=None, saving_name=None, base_column=None, view='summary', mode='base', percent=5, depth=10):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame

        """ Core """
        percentile_range = list()
        percent_ = (percent/100); i = 0
        while True:
            i += 1
            cumulative_percent = round(percent_*i, 4)
            if cumulative_percent >= 1 :
                break
            percentile_range.append(cumulative_percent)
        self.percentiles = percentile_range

        # for Numeric&Categorical Columns
        numerical_table = table[table.columns[(table.dtypes != 'object') & (table.dtypes != 'category')]]
        categorical_table = table[table.columns[(table.dtypes == 'object') | (table.dtypes == 'category')]]
        assert numerical_table.shape[1] >= 1, "This table doesn't even have a single numerical column. Change data-type of columns on table"        
        assert categorical_table.shape[1] >= 1, "This table doesn't even have a single categorical column. Change data-type of columns on table"        
        if base_column is not None:
            assert base_column in numerical_table.columns, "base_column must be have numerical data-type."

        base_percentile_matrix = self.univariate_percentile(priority_frame=numerical_table, save=False, path=path, mode=mode, view='full', percent=percent, visual='off')
        percentile_matrix = pd.DataFrame(columns=base_percentile_matrix.columns.to_list() + ['CohenMeasure', 'CohenMeasureRank', 'ComparisonInstance', 'ComparisonColumn'])
        for numerical_column in base_percentile_matrix['Column']:
            if base_column is None:
                pass
            elif base_column != numerical_column:
                continue
            print(f'* Base Numeric Column : {numerical_column}')
            base_percentile_row = base_percentile_matrix.loc[lambda x: x.Column == numerical_column]
            base_percentile_row.insert(base_percentile_row.shape[1], 'CohenMeasure', 0)
            base_percentile_row.insert(base_percentile_row.shape[1], 'CohenMeasureRank', np.inf)
            base_percentile_row.insert(base_percentile_row.shape[1], 'ComparisonInstance', '-')
            base_percentile_row.insert(base_percentile_row.shape[1], 'ComparisonColumn', '-')
            for categorical_column in categorical_table.columns:
                base_row_frame = pd.DataFrame(columns=base_percentile_matrix.columns.to_list() + ['CohenMeasure', 'CohenMeasureRank', 'ComparisonInstance'])
                for categorical_instance  in categorical_table[categorical_column].value_counts().iloc[:depth].index:
                    appending_table = table.loc[lambda x: x[categorical_column] == categorical_instance]
                    appending_percentile_matrix = self.univariate_percentile(priority_frame=appending_table, save=False, path=path, mode=mode, view='full', percent=percent, visual='off')
                    appending_percentile_matrix.loc[:,'CohenMeasure'] = np.nan
                    appending_percentile_matrix.loc[:,'CohenMeasureRank'] = np.nan
                    appending_percentile_matrix.loc[:,'ComparisonInstance'] = categorical_instance
                    base_row_frame = base_row_frame.append(appending_percentile_matrix.loc[lambda x: x.Column == numerical_column])
                base_row_frame.loc[:,'ComparisonColumn'] = categorical_column
                base_percentile_row = base_percentile_row.append(base_row_frame)
            percentile_matrix = percentile_matrix.append(base_percentile_row)
            
            percentile_matrix_by_cloumn = percentile_matrix.loc[lambda x: x.Column==numerical_column]
            if percentile_matrix_by_cloumn.shape[0] <= 1:
                continue
            base_num = percentile_matrix_by_cloumn.iloc[0]['NumRows' if mode == 'missing' else 'NumRows_EFMV']
            base_mean = percentile_matrix_by_cloumn.iloc[0]['mean']
            base_std = percentile_matrix_by_cloumn.iloc[0]['std']
            
            num = percentile_matrix_by_cloumn['NumRows' if mode == 'missing' else 'NumRows_EFMV']
            mean = percentile_matrix_by_cloumn['mean']
            std = percentile_matrix_by_cloumn['std']
            m = base_mean - mean
            s = np.sqrt(((base_num-1)*base_std + (num-1)*std) / (base_num+num-2))  # the pooled standard deviation

            percentile_matrix.loc[lambda x: x['Column']==numerical_column, 'CohenMeasure'] = m/s
            percentile_matrix.loc[lambda x: x['Column']==numerical_column, 'CohenMeasureRank'] = abs(percentile_matrix.loc[lambda x: x['Column']==numerical_column, 'CohenMeasure']).rank(ascending=False)
        """ Core """

        saving_name = f'{saving_name}_EDA_UnivariateConditionalPercentileAnalysis.csv' if saving_name is not None else 'EDA_UnivariateConditionalAnalysis.csv'
        _csv_saving(percentile_matrix, save, self.path, path, saving_name)

        if view == 'full':
            percentile_matrix = percentile_matrix
        elif view == 'p': # percentils
            percentile_matrix = pd.concat([percentile_matrix['Column'], percentile_matrix.loc[:, 'min':'max'], percentile_matrix.loc[:, 'HighDensityRange' : 'ComparisonColumn']], axis=1)
        elif view == 'ap':
            percentile_matrix = pd.concat([percentile_matrix['Column'], percentile_matrix.loc[:, 'DiffMaxMin':'max'], percentile_matrix.loc[:, 'HighDensityRange': 'ComparisonColumn']], axis=1)
        elif view == 'dp':
            percentile_matrix = pd.concat([percentile_matrix['Column'], percentile_matrix.loc[:, f'min-{percent}%':], percentile_matrix.loc[:, 'ComparisonInstance':'ComparisonColumn']], axis=1)
        elif view == 'adp':
            percentile_matrix = pd.concat([percentile_matrix['Column'], percentile_matrix.loc[:,'DiffMaxMin':'min'], percentile_matrix.loc[:,'max'], percentile_matrix.loc[:, f'min-{percent}%':], percentile_matrix.loc[:, 'ComparisonInstance':'ComparisonColumn']], axis=1)
        elif view == 'result':
            percentile_matrix = pd.concat([percentile_matrix['Column'], percentile_matrix.loc[:, 'HighDensityRange': 'ComparisonColumn']], axis=1)
        elif view == 'summary':
            percentile_matrix = percentile_matrix[['Column', 'ComparisonColumn', 'ComparisonInstance', 'CohenMeasureRank', 'HighDensityRange', 'HighDensityInstance', 'HighDensityMinMaxRangeRatio', 'min', 'max', 'mean', 'std']+[ 'NumRows' if mode=='missing' else 'NumRows_EFMV']].sort_values(['Column', 'ComparisonColumn', 'CohenMeasureRank'])

        return percentile_matrix


    def multivariate_frequency(self, priority_frame=None, save=False, path=None, saving_name=None, base_column=None, column_sequence=None):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame
        
        """ Core """
        assert base_column is not None, 'Set your base_column.'
        
        residual_column_sequence = table.columns.copy().to_list()
        del residual_column_sequence[residual_column_sequence.index(base_column)]
        
        if column_sequence is not None:
            seq_len = len(column_sequence)
            for column in column_sequence:
                del residual_column_sequence[residual_column_sequence.index(column)]
        else:
            seq_len = 0
            
        print('[AILEVER] The Rest Columns, Set the column_sequence of list-type \n', residual_column_sequence)
        if seq_len:
            _, axes = plt.subplots(seq_len+1, 1, figsize=(25, 5*(seq_len+1)))
            sns.heatmap(table.groupby([base_column]).agg('count'), ax=axes[0]).set_title('<base_column : '+base_column+f'> : {pd.unique(table[base_column])}')
            for idx in range(seq_len):
                local_seq = column_sequence[:idx+1]
                table_ = table.groupby([*local_seq, base_column])[base_column].agg('count')

                local_seq.reverse()
                for column in local_seq:
                    table_ = table_.unstack(column).fillna(0)
                    if idx == 0 : common_diff = table_.shape[1]

                sns.heatmap(table_, ax=axes[idx+1]).set_title(f'{column_sequence[idx]} : {pd.unique(table[column_sequence[idx]])[::-1]}')
                split_base = 1
                for column in local_seq:
                    for num_split in range(pd.unique(table[column]).shape[0]*split_base):
                        axes[idx+1].axvline(common_diff*(num_split+1), ls=':', c='yellow')
                        if num_split == pd.unique(table[column]).shape[0]-1:
                            split_base *= pd.unique(table[column]).shape[0]

                for column in pd.unique(table_.columns.to_frame().iloc[:,0]):
                    if idx >= 1:
                        axes[idx+1].axvline(table_.columns.get_loc(column).stop, c='red')

                            
        else:
            _, ax = plt.subplots(1, 1, figsize=(25, 5))
            sns.heatmap(table.groupby([base_column]).agg('count'), ax=ax).set_title('<base_column : '+base_column+f'> : {pd.unique(table[base_column])}')
            
        plt.tight_layout()
        """ Core """


    def information_values(self, priority_frame=None, save=False, path=None, saving_name=None, target_column=None, target_event=None):
        assert target_column is not None, 'Target Column must be defined. Set a target(target_column)  on columns of your table'
        if priority_frame is not None:
            table = priority_frame.copy()
        else:
            table = self.frame.copy()
        table['SELECTION_CRITERION'] = np.random.normal(0,1, size=table.shape[0])

        """ Core """
        all_columns = table.columns.to_list()
        del all_columns[all_columns.index(target_column)]
        columns_except_for_target = all_columns 
        for idx, column in enumerate(columns_except_for_target):
            if idx == 0:
                base_value_counts = table.value_counts(column, ascending=False).reset_index()
                base_value_counts.insert(0, 'Column', column)
                base_value_counts = base_value_counts.values
            else:
                value_counts = table.value_counts(column, ascending=False).reset_index()
                value_counts.insert(0, 'Column', column)
                value_counts = value_counts.values
                base_value_counts = np.r_[base_value_counts, value_counts]

        base = pd.DataFrame(data=base_value_counts, columns=['Column', 'Instance', 'Count'])
        base['Count'] = base['Count'].astype(int)
        base.insert(1, 'NumRows', table.shape[0])
        base.insert(base.shape[1], 'Ratio', base['Count']/base['NumRows'])

        print(f'[AILEVER] Selected target column(target_column) : {target_column}')
        target_instances = pd.unique(table[target_column])
        if target_event is not None:
            event_table = table[table[target_column] == target_event]
            nonevent_table = table[table[target_column] != target_event]
            print(f'[AILEVER] Selected target event(target_event) : {target_event}')
        else:
            event_table = table[table[target_column] == target_instances[0]]
            nonevent_table = table[table[target_column] != target_instances[0]]
            print(f'[AILEVER] Selected target event(target_event) : {target_instances[0]}')
        print(f'[AILEVER] Considerable another target events : {target_instances}')
        base.insert(2, 'NumEventRows', event_table.shape[0])
        base.insert(3, 'NumNonEventRows', nonevent_table.shape[0])

        eventcount_mapper = dict()
        for column in columns_except_for_target:
            for instance in table.value_counts(column, ascending=False).index:
                conditional_event_table = event_table[event_table[column] == instance]
                eventcount_mapper[(column, instance)] = conditional_event_table.shape[0]
        base['EventCount'] = eventcount_mapper.values()
        base = base.assign(NonEventCount=lambda x: x.Count-x.EventCount)
        base = base.assign(EventRatio=lambda x: x.EventCount/x.NumEventRows)
        base = base.assign(NonEventRatio=lambda x: x.NonEventCount/x.NumNonEventRows)
        base = base.assign(AdjEventCount=lambda x: x.EventCount + 0.5)
        base = base.assign(AdjNonEventCount=lambda x: x.NonEventCount + 0.5)
        base = base.assign(AdjEventRate=lambda x: x.AdjEventCount/x.NumEventRows)
        base = base.assign(AdjNonEventRate=lambda x: x.AdjNonEventCount/x.NumNonEventRows)

        basetable_fordist = base[['Column', 'AdjEventRate', 'AdjNonEventRate']]
        event_sum = dict()
        nonevent_sum = dict()
        for column in pd.unique(basetable_fordist['Column']):
            conditional_basetable = basetable_fordist[basetable_fordist['Column'] == column]['AdjEventRate']
            event_sum[column] = conditional_basetable.sum()
            conditional_basetable = basetable_fordist[basetable_fordist['Column'] == column]['AdjNonEventRate']
            nonevent_sum[column] = conditional_basetable.sum()

        conditional_summation_table = base['Column']
        conditional_summation_table = pd.concat([conditional_summation_table, base.Column.apply(lambda x: event_sum[x]).rename('EventSum')], axis=1)
        conditional_summation_table = pd.concat([conditional_summation_table, base.Column.apply(lambda x: nonevent_sum[x]).rename('NonEventSum')], axis=1)
        conditional_summation_table

        base = base.assign(DistAdjEventRate=lambda x: x.AdjEventRate/conditional_summation_table.loc[x.Column.index]['EventSum'])
        base = base.assign(DistAdjNonEventRate=lambda x: x.AdjNonEventRate/conditional_summation_table.loc[x.Column.index]['NonEventSum'])
        base = base.assign(AdjEventWOE=lambda x: np.log(x.DistAdjEventRate/x.DistAdjNonEventRate))
        base = base.assign(AdjNonEventWOE=lambda x: np.log(x.DistAdjNonEventRate/x.DistAdjEventRate))
        base = base.assign(AdjEventInstanceIV=lambda x: (x.DistAdjEventRate - x.DistAdjNonEventRate) * x.AdjEventWOE)
        base = base.assign(AdjNonEventInstanceIV=lambda x: (x.DistAdjNonEventRate - x.DistAdjEventRate) * x.AdjNonEventWOE)

        base['InstanceIVRank'] = np.nan
        for column in pd.unique(base['Column']):
            base.loc[lambda df : df['Column']==column, 'InstanceIVRank'] = base[base['Column']==column].AdjEventInstanceIV.rank(ascending=False)
        

        nonevent_iv_avg = dict()
        base['EventIVSum'] = np.nan
        base['NonEventIVSum'] = np.nan
        base['EventIVAvg'] = np.nan
        base['NonEventIVAvg'] = np.nan
        for column in pd.unique(base['Column']):
            base.loc[lambda df: df['Column'] == column, 'EventIVSum'] = base[base['Column'] == column]['AdjEventInstanceIV'].sum()
            base.loc[lambda df: df['Column'] == column, 'NonEventIVSum'] = base[base['Column'] == column]['AdjNonEventInstanceIV'].sum()
            base.loc[lambda df: df['Column'] == column, 'EventIVAvg'] = base[base['Column'] == column]['AdjEventInstanceIV'].mean()
            base.loc[lambda df: df['Column'] == column, 'NonEventIVAvg'] = base[base['Column'] == column]['AdjNonEventInstanceIV'].mean()

        IVRank_mapper = base.drop_duplicates('Column', keep='first')[['Column', 'EventIVSum']]
        IVRank_mapper['IVSumRank'] = IVRank_mapper.EventIVSum.rank(ascending=False)
        IVRank_mapper = IVRank_mapper[['Column', 'IVSumRank']].set_index('Column').to_dict()['IVSumRank']
        base['IVSumRank'] = base.Column.apply(lambda x: IVRank_mapper[x])
        IVRank_mapper = base.drop_duplicates('Column', keep='first')[['Column', 'EventIVAvg']]
        IVRank_mapper['IVAvgRank'] = IVRank_mapper.EventIVAvg.rank(ascending=False)
        IVRank_mapper = IVRank_mapper[['Column', 'IVAvgRank']].set_index('Column').to_dict()['IVAvgRank']
        base['IVAvgRank'] = base.Column.apply(lambda x: IVRank_mapper[x])

        NumUnique_mapper = pd.Series([], dtype=int)
        for column in table.columns:
            NumUniqueInstance = table[column].value_counts().shape[0]
            NumUnique_mapper[column] = NumUniqueInstance
        base['NumUniqueInstance'] = base['Column'].apply(lambda x: NumUnique_mapper[x])

        self.iv_summary = dict()
        self.iv_summary['result'] = base
        self.iv_summary['column'] = base[['NumRows', 'NumEventRows', 'Column', 'NumUniqueInstance', 'EventIVSum', 'EventIVAvg', 'IVSumRank', 'IVAvgRank']].drop_duplicates().sort_values('IVSumRank')
        self.iv_summary['instance'] = base[['NumRows', 'Column', 'NumEventRows', 'Instance', 'Count', 'EventCount', 'AdjEventWOE', 'AdjEventInstanceIV', 'InstanceIVRank', 'IVSumRank', 'IVAvgRank']].sort_values(['InstanceIVRank', 'Column', 'IVSumRank'])
        """ Core """
        

        saving_name = f'{saving_name}_EDA_InformationValues.csv' if saving_name is not None else 'EDA_InformationValues.csv'
        _csv_saving(base, save, self.path, path, saving_name)
        iv_description = pd.DataFrame(data=['Not useful for prediction',
                                            'Weak predictive Power',
                                            'Medium predictive Power',
                                            'Strong predictive Power',
                                            'Suspicious Predictive Power'],
                                      index=['Less than 0.02',
                                             '0.02 to 0.1',
                                             '0.1 to 0.3',
                                             '0.3 to 0.5',
                                             '>0.5'],
                                      columns=['Variable Predictiveness'])
        iv_description.index.name = 'Information Value'
        print(iv_description)

        return self.iv_summary['column'].set_index('Column').EventIVSum.plot.barh(figsize=(25,7), title='EventIVSum')



    def feature_importance(self):
        pass

    def permutation_importance(self):
        pass




class Counting:
    def __init__(self, frame, path='ExploratoryDataAnalysis'):
        self.frame = frame
        self.path = path

    def CountUniqueValues(self, view='table', save=False, path=None):
        if view == 'table':
            data = list()
            for column in self.frame.columns:
                data.append([column, self.frame.value_counts(column).shape[0]])
            EDAframe = pd.DataFrame(data, columns=['Column', 'NumUniqueInstance'])
            _csv_saving(EDAframe, save, self.path, path, 'CountColumns.csv')

        elif view == 'column':
            EDAframe = pd.DataFrame(columns=['Column', 'Instance', 'Count'])
            for column in self.frame.columns:
                base = self.frame[column].value_counts().reset_index().rename(columns={'index':'Instance', column:'Count'})
                base.insert(0, 'Column', column)
                EDAframe = EDAframe.append(base)
            _csv_saving(EDAframe, save, self.path, path, 'CountInstances.csv')

        return EDAframe


def _cohen_d(data1, data2):
    u1, u2 = np.mean(data1), np.mean(data2)
    n1, n2 = len(data1), len(data2)
    s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))  # the pooled standard deviation
    return (u1 - u2) / s


def _csv_saving(frame, save, default_path, priority_path, name):
    if save:
        if not priority_path:
            if not os.path.isdir(default_path):
                os.mkdir(default_path)
            frame.to_csv(os.path.join(default_path, name))
            print(f'[AILEVER] The file {name} is saved at {default_path}.')

        else:
            if not os.path.isdir(priority_path):
                os.mkdir(priority_path)
            frame.to_csv(os.path.join(priority_path, name))
            print(f'[AILEVER] The file {name} is saved at {priority_path}.')

