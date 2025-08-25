from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-4]
replacement_rates = [1e-5]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m imagenet.train_imagenet',
    'args': [
        {
            'agent': 'cbp',
            'cont_backprop': True,
            'weight_decay': 0.0,
            'num_tasks': 2000,
            'replacement_rate': replacement_rates,
            'lr': lrs,
            'compute_hessian': True,
            'compute_hessian_interval': 10,
            'seed': 2025,
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': False,
            'study_name': exp_name
        }
    ]
}