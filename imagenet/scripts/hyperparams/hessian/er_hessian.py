from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-4]
er_lrs = [1e-5]
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
            'compute_hessian': True,
            'compute_hessian_interval': 10,
            'seed': [2035 + i for i in range(20)],
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': False,
            'study_name': exp_name
        }
    ]
}