from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-3]
er_lrs = [1e-2, 1e-3, 1e-4]
weight_decays = [1e-3, 1e-4, 1e-5]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m permuted_mnist.train_permuted_mnist',
    'args': [
        {
            'agent': 'l2_er',
            'num_features': 1000,
            'lr': lrs,
            'er_lr': er_lrs,
            'weight_decay': weight_decays,
            'seed': [2025 + i for i in range(5)],
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}