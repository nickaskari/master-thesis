import numpy as np
import scipy.interpolate


def calculate_asset_pv(asset_values, durations, yield_curve, interest_rate_shocks, scenario='base'):
    yield_maturities, yield_rates = np.array(
        list(yield_curve.keys())), np.array(list(yield_curve.values()))
    interpolate_yield = scipy.interpolate.interp1d(
        yield_maturities, yield_rates, kind="linear", fill_value="extrapolate")

    shock_maturities = np.array(list(interest_rate_shocks.keys()))
    shock_values = np.array(list(interest_rate_shocks.values()))
    interpolate_shock = scipy.interpolate.interp1d(
        shock_maturities, shock_values, kind="linear", fill_value="extrapolate")

    pv_assets = sum(
        value / ((1 + (interpolate_yield(durations[asset]) * (1 + interpolate_shock(
            durations[asset])) if scenario != 'base' else interpolate_yield(durations[asset]))) ** durations[asset])
        for asset, value in asset_values.items() if asset in durations
    )
    return pv_assets


def calculate_liability_pv(liability_value, liability_duration, yield_curve, interest_rate_shocks, scenario='base'):
    yield_maturities, yield_rates = np.array(
        list(yield_curve.keys())), np.array(list(yield_curve.values()))
    interpolate_yield = scipy.interpolate.interp1d(
        yield_maturities, yield_rates, kind="linear", fill_value="extrapolate")

    shock_maturities = np.array(list(interest_rate_shocks.keys()))
    shock_values = np.array(list(interest_rate_shocks.values()))
    interpolate_shock = scipy.interpolate.interp1d(
        shock_maturities, shock_values, kind="linear", fill_value="extrapolate")

    base_rate = interpolate_yield(liability_duration)
    shocked_rate = base_rate * \
        (1 + interpolate_shock(liability_duration)
         ) if scenario != 'base' else base_rate
    discount_factor = 1 / ((1 + shocked_rate) ** liability_duration)

    pv_liability = liability_value * discount_factor
    return pv_liability


def calculate_interest_risk_scr(asset_values, durations, yield_curve, interest_rate_shocks_up, interest_rate_shocks_down, liability_value, liability_duration):
    pv_assets_base = calculate_asset_pv(
        asset_values, durations, yield_curve, interest_rate_shocks_up, scenario='base')
    pv_assets_up = calculate_asset_pv(
        asset_values, durations, yield_curve, interest_rate_shocks_up, scenario='up')
    pv_assets_down = calculate_asset_pv(
        asset_values, durations, yield_curve, interest_rate_shocks_down, scenario='down')

    pv_liability_base = calculate_liability_pv(
        liability_value, liability_duration, yield_curve, interest_rate_shocks_up, scenario='base')
    pv_liability_up = calculate_liability_pv(
        liability_value, liability_duration, yield_curve, interest_rate_shocks_up, scenario='up')
    pv_liability_down = calculate_liability_pv(
        liability_value, liability_duration, yield_curve, interest_rate_shocks_down, scenario='down')

    delta_bof_up = (pv_assets_up - pv_assets_base) + \
        (pv_liability_up - pv_liability_base)
    delta_bof_down = (pv_assets_down - pv_assets_base) + \
        (pv_liability_down - pv_liability_base)

    return max(abs(delta_bof_up), abs(delta_bof_down))


def calculate_equity_risk_scr(asset_values, equity_shock_global, equity_shock_other):
    equity_scr_global = abs(asset_values.get(
        'global_equity', 0) * equity_shock_global)
    equity_scr_other = abs(asset_values.get(
        'other_equity', 0) * equity_shock_other)
    return np.sqrt(equity_scr_global**2 + equity_scr_other**2 + 2 * 0.75 * equity_scr_global * equity_scr_other)


def calculate_property_risk_scr(asset_values, real_estate_shock):
    return abs(asset_values.get('real_estate', 0) * real_estate_shock)


def calculate_spread_risk_scr(asset_values, durations, spread_shocks_corp):
    return sum(
        asset_values.get(asset_class, 0) *
        spread_shocks_corp[rating] * durations.get(asset_class, 0)
        for asset_class, rating in {'IG_corp_bonds': 'A', 'HY_corp_bonds': 'B'}.items()
    )


def calculate_market_risk_scr(market_scr_interest, market_scr_equity, market_scr_real_estate, market_scr_spread, interest_rate_up):
    A = 0.5 if interest_rate_up else 0
    correlation_matrix = np.array([
        [1.0, A, A, A],
        [A, 1.0, 0.75, 0.75],
        [A, 0.75, 1.0, 0.5],
        [A, 0.75, 0.5, 1.0]
    ])
    risk_vector = np.array(
        [market_scr_interest, market_scr_equity, market_scr_real_estate, market_scr_spread])
    return np.sqrt(np.dot(risk_vector, np.dot(correlation_matrix, risk_vector)))


def calculate_market_scr(
    asset_values, durations, equity_shock_global=-0.39, equity_shock_other=-0.49, real_estate_shock=-0.25, spread_shocks_corp=None,
    interest_rate_shocks_up=None, interest_rate_shocks_down=None, liability_value=0.7, liability_duration=3
):
    if spread_shocks_corp is None:
        spread_shocks_corp = {'AAA': 0.009, 'AA': 0.011,
                              'A': 0.014, 'BBB': 0.025, 'BB': 0.045, 'B': 0.075}

    if interest_rate_shocks_up is None:
        interest_rate_shocks_up = {1: 0.70, 2: 0.70, 3: 0.64, 4: 0.59, 5: 0.55,
                                   6: 0.52, 7: 0.49, 8: 0.47, 9: 0.44, 10: 0.42, 20: 0.26, 90: 0.20}

    if interest_rate_shocks_down is None:
        interest_rate_shocks_down = {1: -0.75, 2: -0.65, 3: -0.56, 4: -0.50, 5: -0.46,
                                     6: -0.42, 7: -0.39, 8: -0.36, 9: -0.33, 10: -0.31, 20: -0.29, 90: -0.20}

    yield_curve = {0: 0.0343, 1: 0.0343, 2: 0.0311, 3: 0.0293, 4: 0.0283,
                   5: 0.0277, 6: 0.0274, 7: 0.0272, 8: 0.0271, 9: 0.0272, 10: 0.0273}

    market_scr_interest = calculate_interest_risk_scr(
        asset_values, durations, yield_curve, interest_rate_shocks_up, interest_rate_shocks_down, liability_value, liability_duration)
    market_scr_equity = calculate_equity_risk_scr(
        asset_values, equity_shock_global, equity_shock_other)
    market_scr_real_estate = calculate_property_risk_scr(
        asset_values, real_estate_shock)
    market_scr_spread = calculate_spread_risk_scr(
        asset_values, durations, spread_shocks_corp)

    interest_rate_up = market_scr_interest == abs(calculate_interest_risk_scr(
        asset_values, durations, yield_curve, interest_rate_shocks_up, interest_rate_shocks_down, liability_value, liability_duration))

    total_market_scr = calculate_market_risk_scr(
        market_scr_interest, market_scr_equity, market_scr_real_estate, market_scr_spread, interest_rate_up)

    return {
        "Market SCR Interest": market_scr_interest,
        "Market SCR Equity": market_scr_equity,
        "Market SCR Real Estate": market_scr_real_estate,
        "Market SCR Spread": market_scr_spread,
        "Total Market SCR": total_market_scr
    }


asset_values = {
    'global_equity': 1/7,
    'other_equity': 1/7,
    'real_estate': 1/7,
    'gov_bonds': 1/7,
    'IG_corp_bonds': 1/7,
    'HY_corp_bonds': 1/7,
    'money_market': 1/7
}

durations = {
    'gov_bonds': 7.29,
    'IG_corp_bonds': 5.90,
    'HY_corp_bonds': 3.14
}

scr_results = calculate_market_scr(asset_values, durations)
for key, value in scr_results.items():
    print(f"{key}: {value:,.3f}")
