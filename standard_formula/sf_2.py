import numpy as np
import pandas as pd

from dotenv.main import load_dotenv
load_dotenv(override=True)
import os
import scipy.interpolate

print("\n")

def calculate_market_scr(
    asset_values,
    durations,
    equity_shock_global=-0.39,
    equity_shock_other=-0.49,
    real_estate_shock=-0.25,  # from CEIOPS-DOC-66/10 page 30
    spread_shocks=None,  #  credit quality steps
    interest_rate_shocks_up=None,  
    interest_rate_shocks_down=None,
    liability_value=float(os.getenv("FRAC_LIABILITIES")),  # Specifies the asset/liability ratio, given that assets are 1.0 
    liability_duration=3 # unsure about this value
):
    if spread_shocks is None:
        spread_shocks = {'gov_bonds': 0.0025, # AA
                         'IG_corp_bonds': 0.0103, # A 
                         'HY_corp_bonds': 0.056} # B
                        # from CEIOPS-DOC-66/10 page 34
                        # these are stress factors F^up(rating_i) from Braun
        
    
    if interest_rate_shocks_up is None:
        interest_rate_shocks_up = {1: 0.70, 2: 0.70, 3: 0.64, 4: 0.59, 5: 0.55, 6: 0.52, 7: 0.49, 
                               8: 0.47, 9: 0.44, 10: 0.42, 11: 0.39, 12: 0.37, 13: 0.35, 14: 0.34, 
                               15: 0.33, 16: 0.31, 17: 0.30, 18: 0.29, 19: 0.27, 20: 0.26, 90: 0.20}

    if interest_rate_shocks_down is None:
        interest_rate_shocks_down = {1: -0.75, 2: -0.65, 3: -0.56, 4: -0.50, 5: -0.46, 6: -0.42, 7: -0.39,
                                 8: -0.36, 9: -0.33, 10: -0.31, 11: -0.30, 12: -0.29, 13: -0.28, 14: -0.28, 
                                 15: -0.27, 16: -0.28, 17: -0.28, 18: -0.28, 19: -0.29, 20: -0.29, 90: -0.20}

    yield_curve = {
        0: 0.0343, 1: 0.0343, 2: 0.0311, 3: 0.0293, 4: 0.0283,
        5: 0.0277, 6: 0.0274, 7: 0.0272, 8: 0.0271, 9: 0.0272, 10: 0.0273
    }


    total_assets = sum(asset_values.values()) 
    print(f"total_assets: {total_assets}")

    fixed_income_assets = {asset: value for asset,
                           value in asset_values.items() if asset in durations}
    total_fixed_income = sum(fixed_income_assets.values())
    print(f"total_fixed_income: {total_fixed_income}")

    if total_fixed_income > 0:
        asset_duration = sum(fixed_income_assets[asset] * durations.get(
            asset, 0) for asset in fixed_income_assets) / total_fixed_income
    else:
        asset_duration = 0
    print(f"asset_duration: {asset_duration}")

    yield_maturities = np.array(list(yield_curve.keys()))
    yield_rates = np.array(list(yield_curve.values()))

    interpolate_yield = scipy.interpolate.interp1d(yield_maturities, yield_rates, kind="linear", fill_value="extrapolate")

    base_rate_liability = interpolate_yield(liability_duration)
    base_discount_factor = 1 / ((1 + base_rate_liability) ** liability_duration)
    
    '''
    avg_interest_rate_up = np.mean([interest_rate_shocks_up[min(interest_rate_shocks_up.keys(), key=lambda x: abs(x - d))]
                                    for d in durations.values()])    
    avg_interest_rate_down = np.mean([interest_rate_shocks_down[min(interest_rate_shocks_down.keys(), key=lambda x: abs(x - d))]
                                      for d in durations.values()])
    # maybe one should interpolate the interest rate shocks, instead of using the nearest one
    print(f"up {avg_interest_rate_up} \n down {avg_interest_rate_down}")
    '''
    shock_maturities = np.array(list(interest_rate_shocks_up.keys()))
    shock_up_values = np.array(list(interest_rate_shocks_up.values()))
    shock_down_values = np.array(list(interest_rate_shocks_down.values()))

    interpolate_up = scipy.interpolate.interp1d(shock_maturities, shock_up_values, kind="linear", fill_value="extrapolate")
    interpolate_down = scipy.interpolate.interp1d(shock_maturities, shock_down_values, kind="linear", fill_value="extrapolate")

    shocked_rate_up = base_rate_liability * (1 + interpolate_up(liability_duration))
    shocked_rate_down = base_rate_liability * (1 + interpolate_down(liability_duration))

    shocked_discount_factor_up = 1 / ((1 + shocked_rate_up) ** liability_duration)
    shocked_discount_factor_down = 1 / ((1 + shocked_rate_down) ** liability_duration)

    pv_liability_base = liability_value * base_discount_factor
    pv_liability_up = liability_value * shocked_discount_factor_up
    pv_liability_down = liability_value * shocked_discount_factor_down

    print(f"Base PV of Liabilities: {pv_liability_base}")
    print(f"Shocked PV Up: {pv_liability_up}")
    print(f"Shocked PV Down: {pv_liability_down}")

    delta_bof_up = (-total_fixed_income * asset_duration * interpolate_up(asset_duration)) + \
        (pv_liability_up -pv_liability_base)


    delta_bof_down = (-total_fixed_income * asset_duration * interpolate_down(asset_duration)) + \
        (pv_liability_down -pv_liability_base)

    # interest rate risk, not sure
    market_scr_interest = max(abs(delta_bof_up), abs(delta_bof_down))

    # Equity risk, good
    equity_scr_global = abs(asset_values.get(
        'global_equity', 0) * equity_shock_global)
    equity_scr_other = abs(asset_values.get(
        'other_equity', 0) * equity_shock_other)

    market_scr_equity = np.sqrt(
        equity_scr_global**2 + equity_scr_other**2 + 2 *
        0.75 * equity_scr_global * equity_scr_other
    )

    # Real estate risk, good
    market_scr_real_estate = abs(asset_values.get('real_estate', 0) * real_estate_shock)

    # Spread risk, good
    market_scr_spread = 0
    for asset_class, shock in spread_shocks.items():
        if asset_class in asset_values:
            market_scr_spread += abs(asset_values[asset_class] * shock * durations.get(asset_class, 0))

    
    interest_rate_up = abs(delta_bof_up) > abs(delta_bof_down)
    A = 0.5 if interest_rate_up else 0

    correlation_matrix = np.array([
        [1.0, A, A, A],
        [A, 1.0, 0.75, 0.75],
        [A, 0.75, 1.0, 0.5],
        [A, 0.75, 0.5, 1.0]
    ])

    risk_vector = np.array(
        [market_scr_interest, market_scr_equity, market_scr_real_estate, market_scr_spread])

    market_scr = np.sqrt(
        np.dot(risk_vector, np.dot(correlation_matrix, risk_vector)))

    return {
        "Market SCR Interest": market_scr_interest,
        "Market SCR Equity": market_scr_equity,
        "Market SCR Real Estate": market_scr_real_estate,
        "Market SCR Spread": market_scr_spread,
        "Total Market SCR": market_scr
    }


equal_weights = 1/7  # 1/7 to Money Market, but that doesnt contribute to SCR
asset_values = {
    'global_equity': equal_weights,
    'other_equity': equal_weights,
    'real_estate': equal_weights,
    'gov_bonds': equal_weights,
    'IG_corp_bonds': equal_weights,
    'HY_corp_bonds': equal_weights,
    'money_market': equal_weights
}


durations = {
    'gov_bonds': 7.29,
    'IG_corp_bonds': 5.90,
    'HY_corp_bonds': 3.14
}

scr_results = calculate_market_scr(asset_values, durations)
for key, value in scr_results.items():
    print(f"{key}: {value:,.2f}")


