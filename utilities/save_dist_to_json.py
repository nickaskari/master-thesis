import json
import os
from dotenv.main import load_dotenv
load_dotenv(override=True)


def save_dist_to_json(model_name, distribution, scr, weights, overwrite=False):
    json_path = "../model_results.json"

    if not overwrite and os.path.exists(json_path):
        with open(json_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data[model_name] = {
        "distribution": distribution,
        "scr": scr,
        "title": model_name,
        "weights": weights,
        "asset_liability_ratio": float(os.getenv("FRAC_LIABILITIES"))
    }

    with open(json_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Results for {model_name} stored in {json_path}")