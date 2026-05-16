from pathlib import Path

exp_name = Path(__file__).stem

lrs = [1e-3]
weight_decays = [1e-5]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m imagenet.train_imagenet',
    'args': [
        {
            'agent': 'laynorm_l2',
            'use_layernorm': True,
            'lr': lrs,
            'weight_decay': weight_decays,
            'compute_hessian': True,
            'compute_hessian_interval': 10,
            'seed': [2035 + i for i in range(20)],
            'n_seeds': 1,
            'num_tasks': 2000,
            'platform': 'gpu',
            'debug': False,
            'study_name': exp_name
        }
    ]
}