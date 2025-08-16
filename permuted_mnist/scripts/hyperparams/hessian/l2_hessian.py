from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-3]
weight_decays = [0.0001]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m permuted_mnist.train_permuted_mnist',
    'args': [
        {
            'agent': 'l2',
            'num_features': 1000,
            'compute_hessian': True,
            'compute_hessian_interval': 10,
            'weight_decay': weight_decays,
            'lr': lrs,
            'seed': 2025,
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}