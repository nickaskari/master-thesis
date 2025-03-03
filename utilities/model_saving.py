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