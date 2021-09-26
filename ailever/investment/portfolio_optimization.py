from ailever.investment import __fmlops_bs__ as fmlops_bs
from .fmlops_loader_system import Loader, parallelize
from .screener import ScreenerModules

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import FinanceDataReader as fdr

loader = Loader()
class PortfolioManagement(ScreenerModule):
    def __init__(self, baskets):
        super(PortfolioManagement, self).__init__(baskets)
        self.initialization()

    def initialization(self):
        self.optimal_portfolios = compose_optimal_portfolio(self.highest_momenta, self.highest_intrinsic_values)
        self.return_matrix = None
        
    def risks(self):
        self.risk_matrix = None
    
    def compose_optimal_portfolio(self, momenta, intrinsic_values):
        pass

    def portfolio_selection(self):
        # TODO : from MCDA
        pass

    def portfolio_optimization(self):
        # TODO : deepdow-like
        pass


class MultiCriteriaDecisionAnalysis:
    def TOPSIS(self):
        # Technique for Order of Preference by Similarity to Ideal Solution
        pass

    def MAUT(self):
        # Multi-Attribute Utility Thoery
        utility_score = None
        return utility_score

    def ELECTRE(self):
        # ELimination Et Choice Translating REality
        pass

    def PROMETHEE(self):
        # Preference ranking organization method for enrichment evaluation
        pass


