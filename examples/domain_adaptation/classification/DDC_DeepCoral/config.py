CFG = {
    'data_path': '/content/Transfer-Learning-Library/examples/domain_adaptation/classification/data/MRSSC/',
    'kwargs': {'num_workers': 4},
    'batch_size': 32,
    'epoch': 10,
    'lr': 1e-3,
    'momentum': .9,
    'log_interval': 10,
    'l2_decay': 0,
    'lambda': 1,
    'backbone': 'resnet50',
    'n_class': 7,
}
