from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-2, 1e-3, 1e-4]
spectral_strengths = [1e-2, 1e-3, 1e-4]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m incremental_cifar.train_incremental_cifar',
    'args': [
        {
            'agent': 'spectral_reg',
            'weight_decay': 0.0,
            'use_spectral_reg': True,
            'spectral_reg_strength': spectral_strengths,
            'lr': lrs,
            'seed': [2025 + i for i in range(5)],
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}