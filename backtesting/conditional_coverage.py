from backtesting.christoffersens_test import christoffersen_independence_test
from backtesting.kupiec_test import kupiec_pof_test
from scipy.stats import chi2

def christoffersen_conditional_coverage_test(failures):
    """
    Performs Christoffersen's Conditional Coverage test by combining
    Kupiec's POF and the Independence test statistics.
    """

    stat_pof, pval_pof = kupiec_pof_test(failures)

    stat_ind, pval_ind = christoffersen_independence_test(failures)

    cc_stat = stat_pof + stat_ind

    cc_pvalue = 1 - chi2.cdf(cc_stat, df=2)

    return cc_stat, cc_pvalue