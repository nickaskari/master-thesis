import json
import os
from dotenv.main import load_dotenv
import numpy as np
import pandas as pd
load_dotenv(override=True)

# Used in model_comparisons.ipynb. That file is at root, so the paths has to be from the root.
json_path = "model_results.json"

def save_results(model_name, distribution, scr, weights, overwrite=False):

    if not overwrite and os.path.exists(json_path):
        with open(json_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    def convert_to_list(obj):
        if isinstance(obj, (np.ndarray, pd.Series)):  # Check if it's an ndarray or Series
            return obj.tolist()
        return obj  # Otherwise, return as is

    distribution = convert_to_list(distribution)
    weights = convert_to_list(weights)

    data[model_name] = {
        "distribution": distribution,
        "scr": scr,
        "title": model_name,
        "weights": weights,
        "asset_liability_ratio": float(os.getenv("FRAC_LIABILITIES", 1.0))  # Default to 1.0 if env var is missing
    }

    try:
        with open(json_path, "w") as file:
            json.dump(data, file, indent=4)

        with open(json_path, "r") as file:
            saved_data = json.load(file)

        if model_name in saved_data:
            print(f"✅ Results for {model_name} successfully stored in {json_path}")
            return True
        else:
            print(f"⚠️ Warning: Data was not saved correctly for {model_name}.")
            return False

    except Exception as e:
        print(f"❌ Error saving results for {model_name}: {e}")
        return False

def is_results_empty():
    if not os.path.exists(json_path):  
        return True
    
    try:
        with open(json_path, "r") as file:
            data = json.load(file)
            return not bool(data)  # Returns True if data is empty
    except (json.JSONDecodeError, ValueError):  # Handle corrupt or empty file
        return True
    
def load_results():
    if not os.path.exists(json_path):  # Check if file exists
        print(f"Warning: {json_path} does not exist. Returning empty dictionary.")
        return {}

    try:
        with open(json_path, "r") as file:
            data = json.load(file)
            return data if data else {}  # Return empty dict if JSON is empty
    except (json.JSONDecodeError, ValueError):  # Handle corrupt or empty file
        print(f"Warning: {json_path} contains invalid JSON. Returning empty dictionary.")
        return {}
    
def append_scenario_result_json(file_path, asset, date, scenarios, overwrite=False):
    """
    Appends (or overwrites) a scenario result to a JSON file.

    Parameters:
      file_path (str): Path to the JSON file.
      asset (str): Asset name.
      date (any): Date for which the scenarios are generated. This will be converted to a string.
      scenarios (np.ndarray): Scenario result array (e.g. shape (n_simulations, 252)).
      overwrite (bool): If True, start from scratch (overwrite the file); 
                        if False, append to existing file.
    """
    file_path = 'rolling_window_results.json'

    # Convert the date to string (if not already)
    date_str = str(date)

    # Convert the scenarios array to a list for JSON storage.
    scenarios_list = scenarios.tolist()

    # If overwrite is True or the file doesn't exist, start with a new dict.
    if overwrite or not os.path.exists(file_path):
        data = {asset: {date_str: scenarios_list}}
    else:
        # Load existing JSON file.
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        # If asset key doesn't exist, create it.
        if asset not in data:
            data[asset] = {}
        # Store/overwrite the scenario result for this date.
        data[asset][date_str] = scenarios_list

    # Write the updated data back to the file.
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_scenario_results_json():
    """
    Loads the scenario results from the given JSON file.
    
    Parameters:
      file_path (str): Path to the JSON file.
    
    Returns:
      dict: The dictionary containing the scenario results.
            Returns an empty dict if the file doesn't exist or is empty.
    """
    file_path = 'rolling_window_results.json'
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return {}
    
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_path}.")
            data = {}
    return data