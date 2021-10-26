import pandas as pd

class Integrated_Loader:
    def __init__(self):
        pass

def Ailever_API(meta_info=True, table:str=None, download=False):
    if table is None:
        meta_table = pd.DataFrame(
                columns=['TABLE', 'DESCRIPTION'], 
                data=[['district0001', 'korea administrative district'], 
                    ])
        return meta_table
    else:
        table = pd.read_csv('https://raw.githubusercontent.com/ailever/dataset/main/{table.upper()}.csv', index_col=0)
        return table
    
