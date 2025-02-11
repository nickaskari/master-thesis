import numpy as np
import pandas as pd

from dotenv.main import load_dotenv
load_dotenv(override=True)
import os

class StandardMarketSCR:

    def __init__(self, weights, durations):
        self.weights = weights 
        self.durations = durations
        self.equity_shock_global = -0.39
        self.equity_shock_other = -0.49
        self.real_estate_shock = -0.25

        self.spread_shocks= {'gov_bonds': 0.0025, # AA
                            'IG_corp_bonds': 0.0103, # A 
                            'HY_corp_bonds': 0.056} 
        
        self.interest_rate_shocks_up = {1: 0.70, 2: 0.70, 3: 0.64, 4: 0.59, 5: 0.55, 6: 0.52, 7: 0.49, 
                               8: 0.47, 9: 0.44, 10: 0.42, 11: 0.39, 12: 0.37, 13: 0.35, 14: 0.34, 
                               15: 0.33, 16: 0.31, 17: 0.30, 18: 0.29, 19: 0.27, 20: 0.26, 90: 0.20}

        self.interest_rate_shocks_down = {1: -0.75, 2: -0.65, 3: -0.56, 4: -0.50, 5: -0.46, 6: -0.42, 7: -0.39,
                                 8: -0.36, 9: -0.33, 10: -0.31, 11: -0.30, 12: -0.29, 13: -0.28, 14: -0.28, 
                                 15: -0.27, 16: -0.28, 17: -0.28, 18: -0.28, 19: -0.29, 20: -0.29, 90: -0.20}


        self.liability_value = float(os.getenv("FRAC_LIABILITIES"))
        self.liability_duration = 3

    def interest_rate_risk(self):
        
        return