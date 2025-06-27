from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-4]
lambda0s = [0.95]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m rlopt.ppo_nonstationary',
    'args': [
        {
            'env': 'slippery_ant',
            'total_steps': 10000000,
            'num_envs': 1,
            'weight_decay': 0.0,
            'change_every': 2000000,
            'cont_backprop': True,
            'compute_hessian_init': True,
            'compute_hessian_end': True,
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