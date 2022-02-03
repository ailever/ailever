from ..logging_system import logger

from scipy import stats
import matplotlib.pyplot as plt

class Hypothesis:
    @staticmethod
    def chi2_contingency(conditional_table, prob=0.95):
        # interpret p-value
        stat, p, dof, expected = stats.chi2_contingency(conditional_table)
        alpha = 1.0 - prob
        logger['analysis'].info('Significance=%.3f, p=%.3f' % (alpha, p))
        if p <= alpha:
            logger['analysis'].info('Dependent (reject H0)')
        else:
            logger['analysis'].info('Independent (fail to reject H0)')
        plt.figure(figsize=(30,5))
        plt.pcolor(conditional_table)
        plt.colorbar()    
        return stat, p, dof, expected


hypothesis = Hypothesis()
