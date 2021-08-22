from .fmlops_loader_system import parallelize

class PortfolioManagement:
    def __init__(self, baskets, period=100):
        self.pdframe = parallelize(baskets=baskets, period=period).pdframe
        self.initialization()

    def initialization(self):
        self.return_matrix = (self.pdframe - self.pdframe.iloc[0])/self.pdframe
        
    def risks(self):
        self.risk_matrix = 
