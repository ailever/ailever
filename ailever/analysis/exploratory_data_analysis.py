import os
import numpy as np
import pandas as pd

class ExploratoryDataAnalysis:
    def __init__(self, frame, save=False, path='ExploratoryDataAnalysis'):
        self.frame = frame
        self.path = path

        if save:
            self._excel()     

    def cleaning(self, priority_frame=None, save=False, path=None):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame

    def frequency(self, priority_frame=None, save=False, path=None):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame

    def transformation(self, priority_frame=None, save=False, path=None):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame

    def selection(self, priority_frame=None, save=False, path=None):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame

    def visualization(self, priority_frame=None, save=False, path=None):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame
    
    def _excel(self, priority_frame=None, save=False, path=None):
        pass

    def table_definition(self, priority_frame=None, save=False, path=None):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame

        table_definition = pd.DataFrame(columns=['NumRows', 'NumColumns', 'NumNumericColumnType', 'NumCategoricalColumnType', ])

        C = 0; N = 0
        for column in table.columns:
            ColumnType = 'Letter' if table[column].dtype == 'object' else 'Number'
            if ColumnType == 'Letter':
                C += 1
            else:
                N += 1

        table_definition = table_definition.append(pd.DataFrame(data=[[table.shape[0], table.shape[1], N, C]], columns=['NumRows', 'NumColumns', 'NumNumericColumnType', 'NumCategoricalColumnType', ]))
        _csv_saving(table_definition, save, self.path, path, 'EDA_TableDefinition.csv')
        return table_definition

    def attributes_specification(self, priority_frame=None, save=False, path=None):
        if priority_frame is not None:
            table = priority_frame
        else:
            table = self.frame

	base_columns = ['Column', 'ColumnType', 'NumUniqueInstance', 'NumMV', 'DataType', 'DataExample', 'MaxInstanceLength']
	attributes_matrix = pd.DataFrame(columns=base_columns)
	
	for column in table.columns:
	    ColumnType = 'Letter' if table[column].dtype == 'object' else 'Number' 
	    NumUniqueInstance = table[column].value_counts().shape[0]
	    MaxInstanceLength = table[column].astype('str').apply(lambda x: len(x)).max()
	    NumMV = table[column].isna().sum()
	    DataType = table[column].dtype.type
	    row = pd.DataFrame(data=[[column, ColumnType, NumUniqueInstance, NumMV, DataType, table[column].iloc[np.random.randint(NumUniqueInstance)], MaxInstanceLength]], columns=base_columns)
	    attributes_matrix = attributes_matrix.append(row)
	    
	attributes_matrix.insert(2, 'NumRows', table.shape[0])
	attributes_matrix.insert(4, 'IdealSymmericCount', table.shape[0]/attributes_matrix['NumUniqueInstance'])
	attributes_matrix.insert(5, 'IdealSymmericRatio', 1/attributes_matrix['NumUniqueInstance'])    
	attributes_matrix.insert(6, 'MVRate', attributes_matrix['NumMV']/table.shape[0])
	attributes_matrix = attributes_matrix.reset_index().drop('index', axis=1)
        _csv_saving(attributes_matrix, save, self.path, path, 'EDA_AttributesSpecification.csv')
	return attributes_matrix




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

