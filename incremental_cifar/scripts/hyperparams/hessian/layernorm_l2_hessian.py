from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-2]
weight_decays = [1e-4]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m incremental_cifar.train_incremental_cifar',
    'args': [
        {
            'agent': 'layernorm_l2',
            'use_layernorm': True,
            'weight_decay': weight_decays,
            'lr': lrs,
            'seed': [2025 + i for i in range(5)],
            'num_tasks': 19,
            'compute_hessian': True,
            'compute_hessian_interval': 1,
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}