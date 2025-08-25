from pathlib import Path

exp_name = Path(__file__).stem

lrs = [0.003]
replacement_rates = [1e-4, 1e-5, 1e-6]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m permuted_mnist.train_permuted_mnist',
    'args': [
        {
            'agent': 'cbp',
            'cont_backprop': True,
            'weight_decay': 0.0,
            'num_features': 1000,
            'lr': lrs,
            'replacement_rate': replacement_rates,
            'seed': [2025 + i for i in range(5)],
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}