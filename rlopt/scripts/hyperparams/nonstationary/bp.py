from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-2, 1e-3, 1e-4]
lambda0s = [0.95]
vf_coeffs = [1.0]

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
            'weight_decay': 0.0,
            'change_every': 2000000,
            'lr': lrs,
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