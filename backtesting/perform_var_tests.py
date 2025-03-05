import numpy as np
from scipy import stats
from backtesting.var_tests.kupiec_test import kupiec_pof_test
from backtesting.var_tests.christoffersens_test import christoffersen_independence_test
from backtesting.var_tests.lopez_loss import lopez_average_loss
from backtesting.var_tests.conditional_coverage import christoffersen_conditional_coverage_test
import os
from dotenv.main import load_dotenv
from utilities.backtesting_plots import calculate_var_threshold
load_dotenv(override=True)


'''
     Lopez’s magnitude loss function - Good for comparing different VaR models
     Joint test - POF and christ. - Known as conditional
     Geometric-VaR backtesting method - seems good (has multiple hypthesis)
'''


''' 
GENERALLY ABOUT THE TESTS:

The statistical methods applied shall test the appropriateness of the probability distribution 
forecast compared not only to loss experience but also to all material new data and information relating thereto.

The model validation process shall include an analysis of the stability of the internal model and in particular
the testing of the sensitivity of the results of the internal model to changes in key underlying assumptions. 
It shall also include an assessment of the accuracy, completeness and appropriateness of the data used by the internal model.

WEBSITE: https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:02009L0138-20190113#id-5ae5e5bb-964e-4047-9bc1-95033a1a9ea1
'''

import os

def perform_var_backtesting_tests(failures, returns, var_forecast, asset_name, generated_returns):
    """
    Runs a series of backtesting tests for a given VaR model:
      - Kupiec POF test (failure frequency)
      - Christoffersen Independence test (clustering of violations)
      - Joint (Conditional Coverage) test (combining the above two)
      - Lopez Average Loss (a magnitude-based loss measure)

    Parameters:
    -----------
    failures : list or np.array
        Binary sequence where 1 indicates a VaR breach.
    returns : list or np.array
        Test period returns.
    var_forecast : float or list or np.array
        Either a single VaR value for all days or an array of VaR forecasts for each day.
    asset_name : str
        The name of the asset for display purposes.
    generated_returns: some array

    Returns:
    --------
    results : dict
        A dictionary containing test statistics, p-values, and conclusions for each test,
        as well as the computed Lopez average loss.
    """
    significance_level = float(os.getenv("SIGNIFICANCE_LEVEL", "0.05"))

    if var_forecast == None:
        var_forecast = [calculate_var_threshold(generated_returns)]

    LR_pof, p_pof = kupiec_pof_test(failures)
    LR_ind, p_ind = christoffersen_independence_test(failures)

    LR_joint, p_joint = christoffersen_conditional_coverage_test(failures)

    avg_lopez_loss = lopez_average_loss(returns, var_forecast)

    if p_pof > significance_level:
        result_pof = f"✅ Passed (p > {significance_level}) - No significant failure pattern detected 🎉"
    else:
        result_pof = f"❌ Failed (p < {significance_level}) - VaR model may be misspecified ⚠️"

    if p_ind > significance_level:
        result_ind = f"✅ Passed (p > {significance_level}) - No clustering of VaR breaches 🎉"
    else:
        result_ind = f"❌ Failed (p < {significance_level}) - VaR breaches are clustered ⚠️"

    if p_joint > significance_level:
        result_joint = f"✅ Passed (p > {significance_level}) - Joint test does not reject model adequacy 🎉"
    else:
        result_joint = f"❌ Failed (p < {significance_level}) - Joint test indicates model misspecification ⚠️"

    print("\n" + "=" * 150)
    print(f"📊 VaR Backtesting Results for {asset_name}")
    print("=" * 150)

    print("\n🔍 Kupiec Proportion of Failures (POF) Test")
    print(f"📝 Likelihood Ratio (LR_pof): {LR_pof:.4f}")
    print(f"📊 p-value: {p_pof:.6f}")
    print(f"🚦 Test Conclusion: {result_pof}")

    print("\n🔍 Christoffersen Independence Test")
    print(f"📝 Likelihood Ratio (LR_ind): {LR_ind:.4f}")
    print(f"📊 p-value: {p_ind:.6f}")
    print(f"🚦 Test Conclusion: {result_ind}")

    print("\n🔍 Joint (Conditional Coverage) Test")
    print(f"📝 Joint Likelihood Ratio (LR_joint): {LR_joint:.4f}")
    print(f"📊 p-value: {p_joint:.6f}")
    print(f"🚦 Test Conclusion: {result_joint}")

    print("\n🔍 Lopez Average Loss")
    print(f"📝 Average Lopez Loss: {avg_lopez_loss:.6f}")
    print("🚦 Lower values indicate fewer or less severe violations.")

    print("=" * 150, "\n")

    return {
        "LR_pof": LR_pof,
        "p_value_pof": p_pof,
        "result_pof": result_pof,
        "LR_ind": LR_ind,
        "p_value_ind": p_ind,
        "result_ind": result_ind,
        "LR_joint": LR_joint,
        "p_value_joint": p_joint,
        "result_joint": result_joint,
        "avg_lopez_loss": avg_lopez_loss
    }
