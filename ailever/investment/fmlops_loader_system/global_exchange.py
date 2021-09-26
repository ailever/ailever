from ailever.investment import __fmlops_bs__ as fmlops_bs
from ..stock_market import MarketInformation
from .parallelizer import Parallelization_Loader

core = fmlops_bs.core['FS1d'] 
PLoader = Parallelization_Loader()

def all_exchanges(markets:list):
    # base stock : 005390
    fdr.DataReader('005390').to_csv(os.path.join(core.path, '005390.csv'))
    MI = MarketInformation()
    market_info = MI.market_info[MI.market_info.Market.apply(lambda x: x in markets)].reset_index().drop('index', axis=1)

    def global_exchange(date='2010-01-01', mode='Close', cut=None, baskets=None):
        nonlocal market_info
        # basket filtering
        if baskets:
            origin_baskets = baskets
            
            # filtering
            symbols = market_info.Symbol.values
            baskets = list(filter(lambda x: x in symbols, baskets))
            serialized_objects = list(map(lambda x: x[:-4], core.listfiles(format='csv')))
            baskets = list(filter(lambda x: x in serialized_objects, baskets))
            baskets = np.array(baskets)
        else:
            baskets = market_info.Symbol.values
            origin_baskets = baskets
            
            # filtering
            serialized_objects = list(map(lambda x: x[:-4], core.listfiles(format='csv')))
            baskets = list(filter(lambda x: x in serialized_objects, baskets))
        
        # Df[0] : Price Dataset
        base_stock = pd.read_csv(os.path.join(core.path,'005930.csv'))
        DTC = PLoader.parallelize(baskets=baskets, path=core.path, base_column=mode, date_column='Date', columns=base_stock.columns.to_list())
        # Df[1] : Stock List
        stock_list = market_info.query(f'Symbol in {baskets}').reset_index().drop('index', axis=1)
        # Df[2] : Exception List
        exception_list = list(filter(lambda x: x not in baskets, origin_baskets))
        # Df[3] : Composite Stock Price Index Lodaer
        financial_market_indicies = dict()
        KIs = ['KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200']
        KI_dict = dict()
        for KI in KIs:
            df = pd.read_csv(os.path.join(core.path, f'{KI}.csv'))
            KI_dict[KI] = df
        financial_market_indicies.update(KI_dict)
        return DTC.ndarray, stock_list, exception_list, financial_market_indicies, mode
    return global_exchange
