import numpy as np

import os
import torch
import numpy as np

def generate_and_save_quarterly_scenarios(cgan_model, conditions, num_scenarios_per_quarter=10000, base_output_folder="generated_cgan_output_test"):
    """
    Generate scenarios for each quarter using a CGAN model, and save the outputs in separate folders.
    
    Parameters:
      cgan_model: The trained CGAN model, assumed to have attributes 'generator', 'latent_dim', 'scaler', and 'cuda'.
      conditions: A numpy array of shape (n_quarters,) or (n_quarters, cond_dim) representing the condition for each quarter.
                  For instance, these could be computed using your create_lagged_quarter_conditions function.
      num_scenarios_per_quarter: Number of scenarios to generate for each quarter.
      base_output_folder: Base directory where folders 'q1', 'q2', ... will be created.
      
    Returns:
      A dictionary where keys are 'q1', 'q2', ... and values are the generated scenarios (NumPy arrays).
    """
    # Create base output folder if it doesn't exist.
    os.makedirs(base_output_folder, exist_ok=True)
    
    batch_size = 1000  # Adjust batch size as needed.
    device = 'cuda' if cgan_model.cuda else 'cpu'
    
    scenarios_by_quarter = {}
    
    # Iterate over each quarter's condition.
    for i, cond in enumerate(conditions):
        quarter_label = f"q{i+1}"
        quarter_folder = os.path.join(base_output_folder, quarter_label)
        os.makedirs(quarter_folder, exist_ok=True)
        
        generated_list = []
        
        # Prepare the condition tensor.
        cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device)
        # Ensure condition is 2D: (1, cond_dim)
        if cond_tensor.ndim == 0:
            cond_tensor = cond_tensor.unsqueeze(0)
        elif cond_tensor.ndim == 1:
            cond_tensor = cond_tensor.unsqueeze(0)
        # For each batch, replicate the condition vector.
        cond_batch = cond_tensor.repeat(batch_size, 1)
        
        cgan_model.generator.eval()  # Set generator to evaluation mode.
        with torch.no_grad():
            # Generate scenarios in batches.
            num_batches = num_scenarios_per_quarter // batch_size
            for _ in range(num_batches):
                z = torch.randn(batch_size, cgan_model.latent_dim, device=device)
                gen_returns = cgan_model.generator(z, cond_batch)
                # Move results to CPU and convert to numpy.
                gen_returns = gen_returns.cpu().numpy()
                # Inverse scale if necessary.
                gen_returns = cgan_model.scaler.inverse_transform(gen_returns)
                generated_list.append(gen_returns)
        
        # Stack all generated batches.
        all_generated = np.vstack(generated_list)
        
        # Save the generated scenarios to a file.
        file_path = os.path.join(quarter_folder, f"generated_returns_{quarter_label}.pt")
        torch.save(torch.tensor(all_generated), file_path)
        print(f"Quarter {i+1}: Generated {num_scenarios_per_quarter} scenarios saved to {file_path}")
        
        scenarios_by_quarter[quarter_label] = all_generated
        
    return scenarios_by_quarter

# Example usage:
# Assume 'cgan_model' is an instance of your CGAN model.
# 'conditions' is a numpy array of quarterly condition values (or vectors) created from your test_returns_df.
# scenarios = generate_and_save_quarterly_scenarios(cgan_model, conditions, num_scenarios_per_quarter=10000)
