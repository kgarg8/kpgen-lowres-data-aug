# lr-4000 - 4000 samples in training set
# v2_sr_aug - T+A followed by T+A (synonym replacement for present kps)
import json, shutil, sys, random, os, numpy as np, pickle, pdb
sys.path.insert(1, '../')
from preprocess_tools.process_utils import jsonl_save
from preprocess_tools.sr_helper import keyphrase_sr_fn
from pathlib import Path
from tqdm import tqdm

NUM_SAMPLES = 4000
K = -1 # number of present keyphrases to replace with synonyms

res = pickle.load(open('../../stage1_processed_data/LDKP3km/LDKP3km_metadata.pkl', 'rb')) # since metadata file is common
vocab2idx = res['vocab2idx']

def process_data(file, path):
    data = []
    with open('../../stage1_processed_data/LDKP3km/'+file) as f:
        for line in f:
            data.append(json.loads(line))

    if 'train' in file:
        subsample = random.sample(range(len(data)), NUM_SAMPLES)
        data = [data[idx] for idx in subsample]
    
    srcs, src_vecs, trgs, trg_vecs = [], [], [], []
    total_count = 0
    count_unchanged = 0
    for i in tqdm(range(len(data))):
        # recreate src, src_vec
        indices = [i for i, x in enumerate(data[i]['src']) if x == '<sep>']
        src     = data[i]['src'][:indices[1]]
        src_vec = data[i]['src_vec'][:indices[1]]

        assert len(src) == len(src_vec)

        # append to dictionary
        srcs.append(src)
        src_vecs.append(src_vec)
        trgs.append(data[i]['trg'])
        trg_vecs.append(data[i]['trg_vec'])
        
        if 'train' in file:
            new_src, new_src_vec, new_trg, new_trg_vec, tc, cu = keyphrase_sr_fn(src, data[i]['trg'], vocab2idx, k=K)
            total_count += tc
            count_unchanged += cu
            srcs.append(new_src)
            src_vecs.append(new_src_vec)
            trgs.append(new_trg)
            trg_vecs.append(new_trg_vec)
    
    print(f'Total count: {total_count}, Count unchanged: {count_unchanged}')
    data_dict = {'src': srcs, 'trg': trgs, 'src_vec': src_vecs, 'trg_vec': trg_vecs}
    # jsonl_save(filepath=path+file.replace('LDKP3km', 'LDKP3km-lr-4000'), data_dict=data_dict)


seeds = [1433, 1896, 1922]

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    dump_folder = '../../stage2_processed_data/LDKP3km-lr-4000/{}/v2_sr_aug_k{}/'.format(seed, K)
    Path(dump_folder).mkdir(parents=True, exist_ok=True)

    process_data('LDKP3km_train.jsonl'   , dump_folder)
    # process_data('LDKP3km_dev_LDKP3km.jsonl' , dump_folder)
    # process_data('LDKP3km_test_LDKP3km.jsonl', dump_folder)
    shutil.copy('../../stage1_processed_data/LDKP3km/LDKP3km_metadata.pkl', dump_folder)  # Copy metadata file as it is
    os.rename(dump_folder+'LDKP3km_metadata.pkl', dump_folder+'LDKP3km-lr-4000_metadata.pkl')