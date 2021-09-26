from ailever.investment import __fmlops_bs__ as fmlops_bs
from .screener import ScreenerModule
from .portfolio_optimizer import SetupInstances, Train

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import FinanceDataReader as fdr


class PortfolioManagement(ScreenerModule):
    def __init__(self, baskets):
        super(PortfolioManagement, self).__init__(baskets)
        self.initialization()

    def initialization(self):
        self.optimal_portfolios = self.compose_optimal_portfolio(self.highest_momenta, self.highest_intrinsic_values)
        self.return_matrix = None
        
    def risks(self):
        self.risk_matrix = None
    
    def compose_optimal_portfolio(self, momenta, intrinsic_values):
        pass

    def portfolio_selection(self):
        # TODO : from MCDA
        pass

    def portfolio_optimization(self, baskets=None, iteration=500):
        if baskets is not None:
            X_ = pd.DataFrame(data=self._portfolio_dataset[:, self.index]).replace([np.inf, -np.inf], np.nan)
        else:
            X_ = pd.DataFrame(data=self._portfolio_dataset[:, self.index]).replace([np.inf, -np.inf], np.nan)
        keeping_columns = X_.dropna(axis=1).columns.to_list()
        dropping_columns = list(filter(lambda x: x not in keeping_columns, X_.columns.to_list()))
        print('* Dropping columns : ', dropping_columns)
        print('* Portfolio : ', self.prllz_df[1].iloc[keeping_columns].Name.to_list())
        X = X_.dropna(axis=1).values

        args = SetupInstances(X=X)
        args['epochs'] = int(iteration)
        weight = Train(*args)
        weight = weight.detach().numpy().squeeze()
        weight = np.where(weight < 0, 0, weight)
        portfolio_weight = pd.DataFrame(data=weight.squeeze(), columns=['StableFactor'], index=self.prllz_df[1].iloc[keeping_columns].Name.to_list()).sort_values(by='StableFactor', ascending=False)
        return portfolio_weight


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


