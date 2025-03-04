import numpy as np
from scipy import stats
from backtesting.kupiec_test import kupiec_pof_test
from backtesting.christoffersens_test import christoffersen_independence_test
import os
from dotenv.main import load_dotenv
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





def perform_var_backtesting_tests(failures, asset_name):
    significance_level = float(os.getenv("SIGNIFICANCE_LEVEL"))

    LR_pof, p_pof = kupiec_pof_test(failures)
    LR_ind, p_ind = christoffersen_independence_test(failures)

    if p_pof > significance_level:
        result_pof = f"✅ Passed (p > {significance_level}) - No significant failure pattern detected 🎉"
    else:
        result_pof = f"❌ Failed (p < {significance_level}) - VaR model may be misspecified ⚠️"

    if p_ind > significance_level:
        result_ind = f"✅ Passed (p > {significance_level}) - No clustering of VaR breaches 🎉"
    else:
        result_ind = f"❌ Failed (p < {significance_level}) - VaR breaches are clustered ⚠️"

    print("\n" + "=" * 50)
    print(f"📊 VaR Backtesting Results for {asset_name}")
    print("=" * 50)

    print("\n🔍 Kupiec Proportion of Failures (POF) Test")
    print(f"📝 Likelihood Ratio (LR_pof): {LR_pof:.4f}")
    print(f"📊 p-value: {p_pof:.6f}")
    print(f"🚦 Test Conclusion: {result_pof}")

    print("\n🔍 Christoffersen Independence Test")
    print(f"📝 Likelihood Ratio (LR_ind): {LR_ind:.4f}")
    print(f"📊 p-value: {p_ind:.6f}")
    print(f"🚦 Test Conclusion: {result_ind}")

    print("=" * 50, "\n")

    return {"LR_pof": LR_pof, "p_value_pof": p_pof, "result_pof": result_pof,
            "LR_ind": LR_ind, "p_value_ind": p_ind, "result_ind": result_ind}
