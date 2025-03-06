import pandas as pd
from backtesting.perform_distribution_tests import perform_distribution_tests
from backtesting.perform_var_tests import perform_var_backtesting_tests
from utilities.backtesting_plots import backtest_var_bof_value, calculate_var_threshold

# Make support for multi VaR

def run_all_tests_on_models(
        models,
        test_returns,
        weights,
        assets_0,
        liabilities_0):
    """
    models: a dictionary (model_name, distribution)
    empirical_returns_rolling: your rolling empirical data
    """
    results_list = []

    for model_name, dist in models.items():

        '''
        dist_results = perform_distribution_tests(
            generated_returns, 
            empirical_returns_rolling, 
            asset_name=model_name, 
            verbose=False  
        )
        '''

        failures = backtest_var_bof_value(model_name, test_returns, dist, weights, assets_0, liabilities_0, confidence_level=0.995,verbose=False)

        var_results = perform_var_backtesting_tests(
            failures=failures, 
            returns=test_returns, 
            var_forecast=[calculate_var_threshold(dist)], 
            asset_name=model_name, 
            generated_returns=dist, 
            verbose=False  
        )

        combined_results = {
            "model_name": model_name,
            **var_results        
        }

        results_list.append(combined_results)

    results_df = pd.DataFrame(results_list)

    return results_df
