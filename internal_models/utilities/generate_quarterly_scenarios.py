import os
import torch
from internal_models.utilities.creating_quarterly_conditions import update_condition_matrix
import numpy as np 

import os
import torch
import numpy as np

def generate_quarterly_scenarios(gan, test_returns, asset_name, num_scenarios_per_condition=10000, quarter_length=63, selected_indices=None):
    """
    Generate scenarios only for selected conditions from the sliding window.
    
    The full condition matrix is computed from historical returns and test returns using a sliding window
    (e.g. 252 conditions for a year if sliding one day at a time). Then, only the conditions at the selected 
    indices are used to generate scenarios. For instance, if selected_indices is [0, 63, 126, 252], then scenarios
    are generated only for those days.
    
    Parameters:
        gan: The CGAN model (assumed to have attributes like returns_series, quarter_length, generator, scaler, etc.)
        test_returns: Test returns data used to compute conditions.
        num_scenarios_per_condition: Number of scenarios to generate for each selected condition.
        quarter_length: Number of days defining a quarter (used in computing conditions).
        selected_indices: List of indices of the conditions to use. If None, defaults to [0, quarter_length, 2*quarter_length, -1].
    
    Returns:
        scenarios_dict: Dictionary where keys are labels (e.g. 'q0', 'q1', ...) and values are the generated scenarios.
    """
    # First, compute all conditions using your existing sliding-window function.
    conditions = update_condition_matrix(gan.returns_series, test_returns, quarter_length)
    total_conditions = len(conditions)
    
    # If no selected indices are provided, default to using day 0, day quarter_length, day 2*quarter_length, and the last day.
    if selected_indices is None:
        selected_indices = [0, quarter_length, 2 * quarter_length]
        if (total_conditions - 1) not in selected_indices:
            selected_indices.append(total_conditions - 1)
    
    print(f"Generating scenarios for conditions at indices: {selected_indices}")
    
    base_output_folder = "generated_CGAN_output_test"
    os.makedirs(base_output_folder, exist_ok=True)
    
    scenarios_dict = {}
    batch_size = 1000  # adjust as needed
    device = 'cuda' if gan.cuda else 'cpu'
    
    gan.generator.eval()
    # Use a sequential counter for the quarter labels (q0, q1, q2, etc.)
    for count, idx in enumerate(selected_indices):
        if idx >= total_conditions:
            print(f"Warning: Requested index {idx} exceeds available conditions ({total_conditions}). Skipping.")
            continue
        
        cond = conditions[idx]
        quarter_label = f"q{count}"
        quarter_folder = os.path.join(base_output_folder, quarter_label)
        os.makedirs(quarter_folder, exist_ok=True)
        
        # Convert the condition to a 2D tensor.
        cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device)
        if cond_tensor.ndim == 0:
            cond_tensor = cond_tensor.unsqueeze(0)
        elif cond_tensor.ndim == 1:
            cond_tensor = cond_tensor.unsqueeze(0)
            
        cond_batch = cond_tensor.repeat(batch_size, 1)
        
        generated_list = []
        with torch.no_grad():
            num_batches = num_scenarios_per_condition // batch_size
            for _ in range(num_batches):
                z = torch.randn(batch_size, gan.latent_dim, device=device)
                gen_returns = gan.generator(z, cond_batch)
                gen_returns = gen_returns.cpu().numpy()
                gen_returns = gan.scaler.inverse_transform(gen_returns)
                generated_list.append(gen_returns)
        all_generated = np.vstack(generated_list)
        
        save_path = os.path.join(quarter_folder, f"generated_returns_{asset_name}_{quarter_label}.pt")
        torch.save(torch.tensor(all_generated), save_path)
        print(f"Condition at index {idx} (labeled {quarter_label}): Generated {num_scenarios_per_condition} scenarios saved to: {save_path}")
        
        scenarios_dict[quarter_label] = all_generated
    
    return scenarios_dict
