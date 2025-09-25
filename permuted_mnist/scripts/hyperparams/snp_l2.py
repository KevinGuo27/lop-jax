from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-2]
weight_decays = [1e-3, 1e-4, 1e-5]
perturb_scales = [1e-4, 1e-5, 1e-6]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m permuted_mnist.train_permuted_mnist',
    'args': [
        {
            'agent': 'snp_l2',
            'num_features': 1000,
            'lr': lrs,
            'to_perturb': True,
            'weight_decay': weight_decays,
            'perturb_scale': perturb_scales,
            'seed': [2025 + i for i in range(5)],
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}