## Column attacher by date
from yahooquery import Ticker
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import monthdelta
import os
import time
from . import _reit_crawler

def reit_subsectors(dir_path="./reit_watch") -> list:
    """
    Return a list of reit subsectors
    """
    if not os.path.isdir(dir_path):
        _reit_crawler()

    date = datetime.today().date() ; date_year = str(date.year) ; date_month = '{:02}'.format(date.month) ; file_path = dir_path+"/RW"+date_year[2:]+date_month+".csv"

    while os.path.isfile(file_path) is False:
        date = date - monthdelta.monthdelta(1) ; date_year = str(date.year) ; date_month = '{:02}'.format(date.month) ; file_path = dir_path+"/RW"+date_year[2:]+date_month+".csv"
        
    print("------------- Loading {} --------------------".format(file_path))

    subsectors = pd.read_csv(file_path)['subsector']
    subsectors = subsectors.drop_duplicates().to_list()
    return subsectors


def reit_tickers(dir_path="./RW") -> dict:
    """
    Return dict {tickers : full-name }
    """
    if not os.path.isdir(dir_path):
        _reit_crawler()

    date = datetime.today().date() ; date_year = str(date.year) ; date_month = '{:02}'.format(date.month) ; file_path = dir_path+"/RW"+date_year[2:]+date_month+".csv"
    while os.path.isfile(file_path) is False:

        date = date - monthdelta.monthdelta(1) ; date_year = str(date.year) ; date_month = '{:02}'.format(date.month) ; file_path = dir_path+"/RW"+date_year[2:]+date_month+".csv"
        
    print("------------- Loading {} --------------------".format(file_path))

    tickers = pd.read_csv(file_path)['Symbol'].to_list()
    names = pd.read_csv(file_path)['Name'].to_list()
    tickers_names = dict(zip(tickers, names))
    return tickers_names

