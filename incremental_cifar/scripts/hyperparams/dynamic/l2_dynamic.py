from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-2]
weight_decays = [1e-4]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m incremental_cifar.train_dynamic_incremental_cifar',
    'args': [
        {
            'agent': 'l2',
            'weight_decay': weight_decays,
            'lr': lrs,
            'seed': [2025 + i for i in range(5)],
            'num_experiments_repeat': 4,
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}