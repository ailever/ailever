import os
import logging

logger_dir = {'ohlcv':'./.ohlvc_log'}                                    


class Loader():
    baskets = None
    from_dir = None
    to_dir = None
    datacore = None

    def init(self, baskets):
        self.baskets = baskets

    def ohlcv_loader(self, baskets, from_dir='./ohlcv', to_dir='./ohlcv', logger_dir=logger_dir['ohlcv'],source='yahooquery'):
        if not baskets:
            baskets = self.baskets
        if not os.path(from_dir):
            os.mkdir(from_dir)
        if not os.path(to_dir):
            os.mkdir(to_dir)

        
class Logger():

    success = None
    failure = None    

