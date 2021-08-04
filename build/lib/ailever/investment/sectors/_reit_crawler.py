## NAREIT https://www.reit.com/sites/default/files/reitwatch/RW2105.pdf 를 기반으로 PDF 및 분류 추출 작업

## 자동 다운로드 및 확인 작업 추가

import tabula
import pandas as pd
import re
from datetime import datetime
import sqlite3
import requests
import monthdelta
import os

## subsectors subject to change

def reit_crawler(dir_path=r"./reit_watch"):

    if not os.path.isdir(dir_path):
        print("new dir: {}".format(dir_path))
        os.mkdir(dir_path)

    else:
        pass
        
    file_date = datetime.today().date() ; year = str(file_date.year);  year = year[2:] ; month = "{:02}".format(file_date.month)
    response = requests.get('https://www.reit.com/sites/default/files/reitwatch/RW' + year + month + '.pdf')

    while response.status_code != 200:

        file_date -= monthdelta.monthdelta(1) ; year = str(file_date.year)[2:] ; month = "{:02}".format(file_date.month)
        response = requests.get('https://www.reit.com/sites/default/files/reitwatch/RW' + year + month + '.pdf')

    path_pdf = dir_path +"/RW" + year + month + ".pdf"
    path_csv = dir_path + "/RW" + year + month + ".csv"

    if not os.path.isfile(path_pdf):
        print("{} downloading".format(path_pdf))

        with open(path_pdf, 'wb') as f:
            f.write(response.content)
    else:     
        print("{} exists. Check {}".format(path_pdf, path_csv))   

    if os.path.isfile(path_csv):
        print("{} exists".format(path_csv))
        df4 = pd.read_csv(path_csv)

    else:
        
        yearinput = int(file_date.year) ; nx_year = yearinput + 1 ; monthinput=file_date.month
        pages = input("pages: ")

        try: 
            dflist = tabula.read_pdf(path_pdf, pages=pages, multiple_tables=True)

        except:
            pass

        line = []

        for i in range(len(dflist)):

            templist=dflist[i].values.tolist()

            for l in range(1, len(templist)):

                if ("Name" not in str(templist[l][0])) and ("AVERAGE" not in str(templist[l][0])):

                    line.append(templist[l][0:5])

        df = pd.DataFrame(line, columns=['Name', 'Symbol', 'Price' ,'52 weeks', 'FFO'] )

        ## Name Column to Series or List

        find_list = list(df.iloc[:,0])

        ## Subsector index finding and sort

        find_list_index = []

        for i in subsectors:
            try:
                temp = [find_list.index(i), i]
                find_list_index.append(temp)
        
            except:
                pass

        find_list_index.sort()    
        ## Subsector column and merge

        col = []

        for l in range(len(find_list_index)):  
            for i in range(len(list(df.iloc[:,0]))): 

                if l == 0:
                    if i < find_list_index[l+1][0]:
                        col.append(find_list_index[l][1])
                elif l > 0 and l < len(find_list_index)-1: 
                    if find_list_index[l][0]<= i < find_list_index[l+1][0] :
                        col.append(find_list_index[l][1])
                else: 
                    if find_list_index[l][0]<= i:
                        col.append(find_list_index[l][1])

        df2 = pd.DataFrame(col, columns=["subsector"])
        df3=pd.merge(df2, df, left_index=True, right_index=True)
        df4=df3.dropna(thresh=5)

        ## Drop unnecessary columns and split columns

        df4 = df4.drop(columns=['Price', '52 weeks'], axis=1)
        df4 = df4.astype(str)

        df4[[str(yearinput)+'FFO', str(nx_year)+'FFO']] = df4['FFO'].str.split(" ", n=1, expand=True)

        df4=df4.drop(columns=['FFO'])

        ## Convert to db

        df4.to_csv(path_csv)
    return df4

    print('Crawling complete')
