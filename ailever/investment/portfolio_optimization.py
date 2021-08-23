from .fmlops_loader_system import parallelize

class PortfolioManagement:
    def __init__(self, baskets, period=100):
        self.pdframe = parallelize(baskets=baskets, period=period).pdframe
        self.initialization()

    def initialization(self):
        self.return_matrix = None
        
    def risks(self):
        self.risk_matrix = None

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


