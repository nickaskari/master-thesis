import os
import torch
from creating_quarterly_conditions import update_condition_matrix
import numpy as np 

def generate_quarterly_scenarios(gan, test_returns, num_scenarios_per_quarter=10000):
    """
    Generate scenarios for each quarter (Q0 to Q4) and store them in separate folders.
    
    The conditions are computed using:
        - Q0: from the last quarter_length days of training (historical) data,
        - Q1-Q4: computed from the test_returns, where each row of test_returns
                represents one quarter's returns.
    
    Parameters:
        test_returns: 2D array of shape (n_quarters, quarter_length) containing out-of-sample test returns.
        num_scenarios_per_quarter: Number of scenarios to generate for each quarter.
        
    Returns:
        scenarios_dict: Dictionary with keys 'q0', 'q1', ... containing generated scenarios.
    """
    # Generate the conditions matrix.
    # historical data is self.returns_series; test_returns is passed in.
    conditions = update_condition_matrix(gan.returns_series, test_returns, quarter_length=gan.quarter_length)
    # conditions shape: (n_quarters+1, 1). Row 0 = Q0 (base condition from historical data),
    # Row 1 = condition from test quarter 1, etc.
    
    base_output_folder = "generated_cgan_output_test"
    os.makedirs(base_output_folder, exist_ok=True)
    
    scenarios_dict = {}
    batch_size = 1000  # or use self.batch_size
    device = 'cuda' if gan.cuda else 'cpu'
    
    gan.generator.eval()
    for q, cond in enumerate(conditions):
        quarter_label = f"q{q}"
        quarter_folder = os.path.join(base_output_folder, quarter_label)
        os.makedirs(quarter_folder, exist_ok=True)
        
        # Convert the condition to tensor, ensuring it is 2D.
        cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device)
        if cond_tensor.ndim == 0:
            cond_tensor = cond_tensor.unsqueeze(0)
        elif cond_tensor.ndim == 1:
            cond_tensor = cond_tensor.unsqueeze(0)
        # Repeat for a batch.
        cond_batch = cond_tensor.repeat(batch_size, 1)
        
        generated_list = []
        with torch.no_grad():
            num_batches = num_scenarios_per_quarter // batch_size
            for _ in range(num_batches):
                z = torch.randn(batch_size, gan.latent_dim, device=device)
                gen_returns = gan.generator(z, cond_batch)
                gen_returns = gen_returns.cpu().numpy()
                # Inverse scale the generated returns.
                gen_returns = gan.scaler.inverse_transform(gen_returns)
                generated_list.append(gen_returns)
        all_generated = np.vstack(generated_list)
        
        # Save the generated scenarios to a file.
        save_path = os.path.join(quarter_folder, f"generated_returns_{quarter_label}.pt")
        torch.save(torch.tensor(all_generated), save_path)
        print(f"Quarter {q}: Generated {num_scenarios_per_quarter} scenarios saved to: {save_path}")
        
        scenarios_dict[quarter_label] = all_generated
    
    return scenarios_dict