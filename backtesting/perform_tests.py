import numpy as np
from scipy import stats
from backtesting.kupiec_test import kupiec_pof_test
from backtesting.christoffersens_test import christoffersen_independence_test
import os
from dotenv.main import load_dotenv
load_dotenv(override=True)

def perform_var_backtesting_tests(failures, asset_name):
    significance_level = float(os.getenv("SIGNIFICANCE_LEVEL"))

    LR_pof, p_pof = kupiec_pof_test(failures)
    LR_ind, p_ind = christoffersen_independence_test(failures)

    if p_pof > significance_level:
        result_pof = "✅ Passed (p > 0.05) - No significant failure pattern detected 🎉"
    else:
        result_pof = "❌ Failed (p < 0.05) - VaR model may be misspecified ⚠️"

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
