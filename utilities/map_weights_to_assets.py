# Used for standard formula calculation

import numpy as np
import pandas as pd



asset_values = {
    "global_equity": "MSCIWORLD",
    "other_equity": "PE",
    "real_estate": "REEL",
    "gov_bonds": "GOV",
    "IG_corp_bonds": "IG",
    "HY_corp_bonds": "HY",
    "money_market": "EONIA"
}

def map_weights_to_assets(returns_df, weights):
    """
    returns_df: DataFrame with columns that match the values in asset_values.
    asset_values: dict mapping your internal asset keys to DataFrame column names.
    weights: 1D array of length = number of columns in returns_df.
    """
    weights_dict = {}
    
    # Make sure we have the same number of columns as weights
    if len(returns_df.columns) != len(weights):
        raise ValueError("Length of weights must match the number of DataFrame columns.")

    # For each key in asset_values, find the corresponding column name
    # and match it to the correct weight from the array.
    for asset_key, column_name in asset_values.items():
        col_index = returns_df.columns.get_loc(column_name)
        weights_dict[asset_key] = weights[col_index]
    
    return weights_dict

