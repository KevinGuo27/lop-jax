from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-2]
replacement_rates = [1e-6]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m incremental_cifar.train_incremental_cifar',
    'args': [
        {
            'agent': 'cbp',
            'cont_backprop': True,
            'weight_decay': 0.0,
            'lr': lrs,
            'num_tasks': 19,
            'compute_hessian': True,
            'compute_hessian_interval': 1,
            'replacement_rate': replacement_rates,
            'seed': [2025 + i for i in range(5)],
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}