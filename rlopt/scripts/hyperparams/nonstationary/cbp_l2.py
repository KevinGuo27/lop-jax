from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-4]
lambda0s = [0.95]
vf_coeffs = [1.0]
replacement_rates = [1e-5, 1e-6, 1e-7]
weight_decays = [1e-3, 1e-4, 1e-5]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m rlopt.ppo_nonstationary',
    'args': [
        {
            'env': 'slippery_ant',
            'total_steps': 10000000,
            'num_envs': 1,
            'num_minibatches': 128,
            'update_epochs': 10,
            'num_steps': 2048,
            'vf_coeff': vf_coeffs,
            'replacement_rate': replacement_rates,
            'maturity_threshold': 10000,
            'weight_decay': weight_decays,
            'change_every': 2000000,
            'cont_backprop': True,
            'lr': lrs,
            'beta_1': 0.99,
            'beta_2': 0.99,
            'lambda0': lambda0s,
            'seed': 2025,
            'n_seeds': 5,
            'platform': 'gpu',
            'steps_log_freq': 4,
            'update_log_freq': 8,
            'debug': True,
            'study_name': exp_name
        }
    ]
}