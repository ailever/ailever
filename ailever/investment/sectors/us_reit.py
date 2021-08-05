from .._base_transfer import DataTransferCore
from typing import Optional, Any, Union, Callable, Iterable
import pandas as pd
import numpy as np
from datetime import datetime
import monthdelta
import os
import time
import tabula
import re
import requests

class us_reit():

    datacore = DataTransferCore()
    date = None
    dir_path = None
    subsector = None


    def __init__(self, dir_path:str='./reit_watch', pages='36-41', source='web', subsector="*"):
        
        self.date = datetime.today().date()
        self.dir_path = dir_path
        self.reit_crawler(self.dir_path, pages=pages, source=source)
        self.reit_subsectors()
        self.reit_get_tickers(subsector)

    def reit_subsectors(self):
        
        self.subsector = list(set(list(self.datacore.dict.values())))
        return self
    
    def reit_get_tickers(self, subsector="*"):

        if subsector == "*":
            """Get all tickers"""
            self.datacore.list = list(set(list((self.datacore.dict.keys()))))
            return self
        
        self.datacore.list = [ticker for ticker, subsec in self.datacore.dict.items() if subsec == subsector]
        return self

    def reit_crawler(self, dir_path=False, pages=False, source=False):
        """
        NAREIT https://www.reit.com/sites/default/files/reitwatch/RW2105.pdf 를 기반으로 PDF 및 분류 추출 작업
        """
        if not dir_path:
            dir_path = self.dir_path
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        date = self.date ; file_date = datetime.strftime(date, "%y%d")

        if source=='local':
            path_csv = os.path.join(dir_path, file_date+ ".csv")
            while not os.path.isfile(path_csv):
                date -= monthdelta.monthdelta(1) ; file_date = datetime.strftime(date,"%y%d")
                path_csv = os.path.join(dir_path, file_date + ".csv")

            from_csv = pd.read_csv(path_csv)
            
            self.datacore.pdframe = from_csv
            self.datacore.dict = dict(zip(from_csv['ticker'].tolist(), from_csv['subsector'].tolist()))

            return self

        if source=='web':       
            response = requests.get('https://www.reit.com/sites/default/files/reitwatch/RW' + file_date + '.pdf')
            while response.status_code != 200:
                date -= monthdelta.monthdelta(1) ; file_date = datetime.strftime(date,"%y%d")
                response = requests.get('https://www.reit.com/sites/default/files/reitwatch/RW' + file_date + '.pdf')

            path_pdf = os.path.join(dir_path, file_date + ".pdf")
            path_csv = os.path.join(dir_path, file_date+ ".csv")

            if not os.path.isfile(path_pdf):
                print("{} downloading".format(path_pdf))
                with open(path_pdf, 'wb') as f:
                    f.write(response.content)
        
            if os.path.isfile(path_csv):
                print("{} loading".format(path_csv))
                df = pd.read_csv(path_csv)
                self.datacore.pdframe = df
                return self.datacore.pdframe

            try: 
                from_pdf = tabula.read_pdf(path_pdf, pages=pages, multiple_tables=True)

            except:
                print("--------------------------------{} are not loadable---------------------------".format(path_pdf))
                date -= monthdelta(1) ; file_date = datetime.strftime(date,"%y%d") ; path_csv = os.path,join(dir_path, file_date+ ".csv")
                print("{} loading".format(path_csv))
                df = pd.read_csv(path_csv)
                self.datacore.pdframe = df
                return self.datacore.pdframe

            tables = pd.concat(from_pdf, axis=0)


            mask = (tables.iloc[:, 0] != 'Name') & (tables.iloc[:, 0] != "AVEREAE") & (tables.iloc[:,0] !="OVERALL AVERAGE")
            tables_striped = tables[mask].iloc[:, 0:2]
            tables_striped.columns = ['name', 'ticker']
            table_striped = tables_striped.reset_index(drop=True)
            table_striped = table_striped.applymap(lambda x: x if type(x)==str else 'mask')
            table_striped = table_striped[table_striped['name']!='mask']

            subsector_frame = table_striped[table_striped['ticker']=='mask']
            subsector_idx = list(zip(subsector_frame.index, subsector_frame['name']))
            table_striped['subsector'] = 0
            for idx, value in enumerate(subsector_idx):
                try:
                    if len(subsector_idx) -1 != idx:                
                        i = subsector_idx[idx][0]
                        j = subsector_idx[idx+1][0]
                        table_striped['subsector'].iloc[i:j] = subsector_idx[idx][1]
                    else:
                        i = subsector_idx[idx][0]
                        table_striped['subsector'].iloc[i:] = subsector_idx[idx][1]
                except Exception as e:
                    print(e)

            table_striped = table_striped[table_striped['ticker']!='mask']
            table_striped = table_striped[['ticker','subsector', 'name']]
            table_striped = table_striped.reset_index(drop=True)
            table_striped.to_csv(path_csv)

            self.datacore.pdframe = table_striped
            self.datacore.dict = dict(zip(table_striped['ticker'].tolist(), table_striped['subsector'].tolist()))

        return self

if __name__ == "__main__":

    a = us_reit()
    b = a.reit_crawler()
