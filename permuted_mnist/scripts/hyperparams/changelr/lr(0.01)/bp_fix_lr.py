from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-2]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m permuted_mnist.train_permuted_mnist',
    'args': [
        {
            'agent': 'bp',
            'weight_decay': 0.0,
            'num_features': 1000,
            'lr': lrs,
            'seed': [2025 + i for i in range(5)],
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}