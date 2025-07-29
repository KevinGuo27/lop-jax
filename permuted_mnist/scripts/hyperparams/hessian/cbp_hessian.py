from pathlib import Path

exp_name = Path(__file__).stem

lrs = [0.001]
replacement_rates = [1e-6]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m permuted_mnist.train_permuted_mnist',
    'args': [
        {
            'agent': 'cbp',
            'cont_backprop': True,
            'weight_decay': 0.0,
            'num_features': 100,
            'lr': lrs,
            'compute_hessian': True,
            'compute_hessian_interval': 10,
            'replacement_rate': replacement_rates,
            'seed': 2025,
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}