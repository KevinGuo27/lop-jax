from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-2, 1e-3, 1e-4]
er_lrs = [1e-3, 1e-4, 1e-5]
er_batchs = [12]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m imagenet.train_imagenet',
    'args': [
        {
            'agent': 'er',
            'num_tasks': 2000,
            'lr': lrs,
            'er_lr': er_lrs,
            'er_batch': er_batchs,
            'weight_decay': 0.0,
            'seed': [2025 + i for i in range(10)],
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': False,
            'study_name': exp_name
        }
    ]
}