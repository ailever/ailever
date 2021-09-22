from .assets import Asset
from .stock_market import market_information
from .screener import Screener
from .portfolio_optimization import PortfolioManagement
from .fmlops_models import Forecaster
from .fmlops_loader_system import Loader, PLoader, Preprocessor
from . import sectors

prllz_loader = PLoader.prllz_loader
parallelize = PLoader.parallelize


def Management(system):
    return locals()[system]
