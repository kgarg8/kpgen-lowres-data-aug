# lr-8000 - 8000 samples in training set
# v2 - title <sep> abstract (w/o truncation)
import json, shutil, sys, random, os, pdb
sys.path.insert(1, '../')
from preprocess_tools.process_utils import jsonl_save
from pathlib import Path

NUM_SAMPLES = 8000

def process_data(file, path):
    data = []
    with open('../../stage1_processed_data/KPTimes/'+file) as f:
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
    jsonl_save(filepath=path+file.replace('KPTimes', 'KPTimes-lr-8000'), data_dict=data_dict)

seeds = [1433, 1896, 1922]

for seed in seeds:
    random.seed(seed)
    dump_folder = '../../stage2_processed_data/KPTimes-lr-8000/{}/v2/'.format(seed)
    Path(dump_folder).mkdir(parents=True, exist_ok=True)

    process_data('KPTimes_train.jsonl'   , dump_folder)
    process_data('KPTimes_dev_KPTimes.jsonl' , dump_folder)
    process_data('KPTimes_test_KPTimes.jsonl', dump_folder)
    shutil.copy('../../stage1_processed_data/KPTimes/KPTimes_metadata.pkl', dump_folder)  # Copy metadata file as it is
    os.rename(dump_folder+'KPTimes_metadata.pkl', dump_folder+'KPTimes-lr-8000_metadata.pkl')