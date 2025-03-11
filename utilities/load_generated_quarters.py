import os
import torch

def load_all_generated_quarters(asset_name, quarterly=False, test=False):
    if test:
        load_dir = 'generated_CGAN_output_test'
    else:
        load_dir = 'generated_CGAN_output'

    if quarterly:
        gen_returns = []
        for q in range(4):  # For quarters q0, q1, q2, q3
            quarter_dir = os.path.join(load_dir, f'q{q}')
            quarter_num = f'q{q}'
            file_path = os.path.join(quarter_dir, f'generated_returns_{asset_name}_{quarter_num}.pt')
            returns = torch.load(file_path)
            gen_returns.append(returns)
    else:
        file_path = os.path.join(load_dir, f'generated_returns_{asset_name}_final_scenarios.pt')
        gen_returns = torch.load(file_path)

    return gen_returns