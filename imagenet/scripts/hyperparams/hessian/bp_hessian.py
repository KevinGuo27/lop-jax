from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-4]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m imagenet.train_imagenet',
    'args': [
        {
            'agent': 'bp',
            'weight_decay': 0.0,
            'lr': lrs,
            'seed': 2025,
            'num_tasks': 2000,
            'compute_hessian': True,
            'compute_hessian_interval': 10,
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': False,
            'study_name': exp_name
        }
    ]
}