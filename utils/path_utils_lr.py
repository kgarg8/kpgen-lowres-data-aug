import pickle
from pathlib import Path

def load_paths(args, time=0):
    metadata_path = Path('processed_data/{}/{}/{}_metadata.pkl'.format(args.lr_seed, args.version, args.dataset))
    with open(metadata_path, 'rb') as fp:
        metadata = pickle.load(fp)
    
    root_dataset = {'LDKP10km': 'LDKP10km', 
    'LDKP10km-lr-1000': 'LDKP10km',
    'LDKP10km-lr-2000': 'LDKP10km',
    'LDKP10km-lr-4000': 'LDKP10km',
    'LDKP10km-lr-8000': 'LDKP10km',
    'LDKP10km-lr-20000': 'LDKP10km',
    'LDKP3km-lr-body-20000': 'LDKP3km'}

    dev_keys  = [args.dataset]
    test_keys  = [args.dataset]

    paths = {}
    paths['train'] = Path('processed_data/{}/{}/{}_train.jsonl'.format(args.lr_seed, args.version, args.dataset))
    paths['dev']   = {key: Path('processed_data/{}/{}/{}_dev_{}.jsonl'.format(args.lr_seed, args.version, args.dataset, key)) for key in dev_keys}
    paths['test']  = {}
    for key in test_keys:
        if key in ['inspec', 'kp20k', 'krapivin', 'nus', 'semeval']:
            paths['test'].update({key: Path('stage1_processed_data/kp_datasets/{}_test_{}.jsonl'.format(root_dataset[args.dataset], key))})
        else:
            paths['test'].update({key: Path('processed_data/{}/{}/{}_test_{}.jsonl'.format(args.lr_seed, args.version, args.dataset, key))})

    test_flag = "_test" if args.test else ""
    
    paths["verbose_log_path"] = Path("experiments/{}/{}/logs/{}_{}_{}/{}_verbose_logs{}.txt".format(args.lr_seed, args.version, args.dataset, args.model, args.model_type, time, test_flag))
    paths["log_path"]         = Path("experiments/{}/{}/logs/{}_{}_{}/{}_logs{}.txt".format(args.lr_seed, args.version, args.dataset, args.model, args.model_type, time, test_flag))
    paths["stats_path"]       = Path("experiments/{}/{}/logs/{}_{}_{}/{}_stats{}.txt".format(args.lr_seed, args.version, args.dataset, args.model, args.model_type, time, test_flag))

    Path('experiments/{}/{}/checkpoints'.format(args.lr_seed, args.version)).mkdir(parents=True, exist_ok=True)
    Path('experiments/{}/{}/logs/{}_{}_{}'.format(args.lr_seed, args.version, args.dataset, args.model, args.model_type)).mkdir(parents=True, exist_ok=True)
    Path('experiments/{}/{}/saved_weights/{}_{}_{}'.format(args.lr_seed, args.version, args.dataset, args.model, args.model_type)).mkdir(parents=True, exist_ok=True)

    if not args.checkpoint:
        with open(paths['verbose_log_path'], 'w+') as fp:
            pass
        with open(paths['log_path'], 'w+') as fp:
            pass
        with open(paths['stats_path'], 'w+') as fp:
            pass
    
    checkpoint_paths = {'infer_checkpoint_path': Path('experiments/{}/{}/saved_weights/{}_{}_{}/{}.pt'.format(args.lr_seed, args.version, args.dataset, args.model, args.model_type, time)),
                        'temp_checkpoint_path': Path('experiments/{}/{}/checkpoints/{}_{}_{}.pt'.format(args.lr_seed, args.version, args.dataset, args.model, args.model_type))}

    return paths, checkpoint_paths, metadata