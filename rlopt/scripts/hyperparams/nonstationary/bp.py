from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5, 2.5e-6]
lambda0s = [0.1, 0.5, 0.7, 0.9, 0.95]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m rlopt.ppo_nonstationary',
    'args': [
        {
            'env': 'slippery_ant',
            'total_steps': 10000000,
            'num_envs': 1,
            'num_minibatches': 16,
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