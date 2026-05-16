from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-2]
spectral_strengths = [1e-2]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m imagenet.train_imagenet',
    'args': [
        {
            'agent': 'spectral_reg',
            'use_spectral_reg': True,
            'lr': lrs,
            'spectral_reg_strength': spectral_strengths,
            'weight_decay': 0.0,
            'compute_hessian': True,
            'compute_hessian_interval': 10,
            'seed': [2035 + i for i in range(20)],
            'n_seeds': 1,
            'num_tasks': 2000,
            'platform': 'gpu',
            'debug': False,
            'study_name': exp_name
        }
    ]
}

