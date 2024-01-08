# lr-2000 - 2000 samples in training set
# v2 - title <sep> abstract (w/o truncation)
import json, shutil, sys, random, os, pdb
sys.path.insert(1, '../')
from preprocess_tools.process_utils import jsonl_save
from pathlib import Path

NUM_SAMPLES = 2000

def process_data(file, path):
    data = []
    with open('../../stage1_processed_data/LDKP3km/'+file) as f:
        for line in f:
            data.append(json.loads(line))

    if 'train' in file:
        subsample = random.sample(range(len(data)), NUM_SAMPLES)
        data = [data[idx] for idx in subsample]
    
    srcs, src_vecs, trgs, trg_vecs = [], [], [], []

    for i in range(len(data)):
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
        
    data_dict = {'src': srcs, 'trg': trgs, 'src_vec': src_vecs, 'trg_vec': trg_vecs}
    jsonl_save(filepath=path+file.replace('LDKP3km', 'LDKP3km-lr-2000'), data_dict=data_dict)

seeds = [1433, 1896, 1922]

for seed in seeds:
    random.seed(seed)
    dump_folder = '../../stage2_processed_data/LDKP3km-lr-2000/{}/v2/'.format(seed)
    Path(dump_folder).mkdir(parents=True, exist_ok=True)

    process_data('LDKP3km_train.jsonl'   , dump_folder)
    process_data('LDKP3km_dev_LDKP3km.jsonl' , dump_folder)
    process_data('LDKP3km_test_LDKP3km.jsonl', dump_folder)
    shutil.copy('../../stage1_processed_data/LDKP3km/LDKP3km_metadata.pkl', dump_folder)  # Copy metadata file as it is
    os.rename(dump_folder+'LDKP3km_metadata.pkl', dump_folder+'LDKP3km-lr-2000_metadata.pkl')