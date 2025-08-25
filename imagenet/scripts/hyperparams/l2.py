from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-2, 1e-3, 1e-4]
weight_decays = [1e-3, 1e-4, 1e-5]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m imagenet.train_imagenet',
    'args': [
        {
            'agent': 'l2',
            'lr': lrs,
            'weight_decay': weight_decays,
            'seed': [2025 + i for i in range(10)],
            'n_seeds': 1,
            'num_tasks': 2000,
            'platform': 'gpu',
            'debug': False,
            'study_name': exp_name
        }
    ]
}