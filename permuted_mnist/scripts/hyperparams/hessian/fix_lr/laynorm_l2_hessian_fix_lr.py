from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-2]
weight_decays = [1e-5]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m permuted_mnist.train_permuted_mnist',
    'args': [
        {
            'agent': 'laynorm_l2',
            'num_features': 1000,
            'weight_decay': weight_decays,
            'compute_hessian': True,
            'compute_hessian_interval': 10,
            'lr': lrs,
            'use_layernorm': True,
            'seed': [2025 + i for i in range(5)],
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}