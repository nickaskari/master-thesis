import numpy as np
from scipy import stats
from backtesting.var_tests.kupiec_test import kupiec_pof_test
from backtesting.var_tests.christoffersens_test import christoffersen_independence_test
from backtesting.var_tests.lopez_loss import lopez_average_loss
from backtesting.var_tests.conditional_coverage import christoffersen_conditional_coverage_test
from backtesting.var_tests.balanced_scr_loss import balanced_scr_loss
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

def perform_var_backtesting_tests(failures, returns, var_forecast, asset_name, generated_returns, verbose=True, portfolio=False, weights=None, bof=None):
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

    if portfolio:
        avg_lopez_loss = lopez_average_loss(bof, var_forecast)
        balanced_loss = balanced_scr_loss(bof, var_forecast)
    else:
        avg_lopez_loss = lopez_average_loss(returns, var_forecast)
        balanced_loss = None


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

    if verbose:
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

        print("\n🔍 Balanced SCR Loss")
        print(f"📝 Loss: {avg_lopez_loss:.6f}")
        print("🚦 A higher value is worse.")

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
        "avg_lopez_loss": avg_lopez_loss,
        "balanced_scr_loss" : balanced_loss
    }

def evaluate_risk_metrics(scr_values, realized_delta_bof_values, alpha_values=None):
   import numpy as np
   import pandas as pd
   from scipy import stats
   
   if alpha_values is None:
       alpha_values = [0.25, 0.5, 0.75]
   
   significance_level = float(os.getenv("SIGNIFICANCE_LEVEL", "0.05"))
   alpha_values = [0.25, 0.5, 0.75]
   realized_delta_bof_values = np.array(realized_delta_bof_values, dtype=np.float64)
   scr_values = np.array(scr_values, dtype=np.float64)

   if len(realized_delta_bof_values) != len(scr_values):
       min_length = min(len(realized_delta_bof_values), len(scr_values))
       
       if len(realized_delta_bof_values) > min_length:
           realized_delta_bof_values = realized_delta_bof_values[:min_length]
       
       if len(scr_values) > min_length:
           scr_values = scr_values[:min_length]

   breaches = np.where(realized_delta_bof_values < scr_values, 1, 0)
   breach_rate = np.mean(breaches)

   results = {
       "Metric": [],
       "Value": [],
       "Description": []
   }

   for alpha in alpha_values:
       bsl = balanced_scr_loss(realized_delta_bof_values, scr_values, alpha)
       results["Metric"].append(f"Balanced SCR Loss (α={alpha})")
       results["Value"].append(bsl)
       results["Description"].append(f"Penalty for {'capital inefficiency' if alpha > 0.5 else 'under-conservatism' if alpha < 0.5 else 'balanced'}")

   print("breaches", breaches)
   kupiec_stat, kupiec_pval = kupiec_pof_test(breaches)
   results["Metric"].extend(["Kupiec POF Test Statistic", "Kupiec POF Test p-value"])
   results["Value"].extend([kupiec_stat, kupiec_pval])
   results["Description"].extend(["Kupiec test statistic", "Tests if breach frequency matches expected"])

   ind_stat, ind_pval = christoffersen_independence_test(breaches)
   results["Metric"].extend(["Indep. Test Statistic", "Indep. Test p-value"])
   results["Value"].extend([ind_stat, ind_pval])
   results["Description"].extend(["Independence test statistic", "Tests if breaches are independent"])

   cc_stat, cc_pval = christoffersen_conditional_coverage_test(breaches)
   results["Metric"].extend(["Cond. Coverage Statistic", "Cond. Coverage p-value"])
   results["Value"].extend([cc_stat, cc_pval])
   results["Description"].extend(["Conditional coverage test statistic", "Combined test of frequency and independence"])

   lopez_loss = lopez_average_loss(realized_delta_bof_values, scr_values)
   results["Metric"].append("Lopez Average Loss")
   results["Value"].append(lopez_loss)
   results["Description"].append("Quadratic loss function for VaR breaches")

   results["Metric"].append("Breach Rate")
   results["Value"].append(breach_rate)
   results["Description"].append("Proportion of time when Delta BOF < SCR")

   df = pd.DataFrame(results)

   test_metrics = ["Kupiec POF Test p-value", "Indep. Test p-value", "Cond. Coverage p-value"]

   df['Pass/Fail'] = df.apply(
       lambda row: '✅' if (row['Metric'] in test_metrics and row['Value'] > significance_level) 
                   else '❌' if row['Metric'] in test_metrics 
                   else '', 
       axis=1
   )

   df['Value'] = df['Value'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

   return df