# lr-2000 - 2000 samples in training set
# v2_sr_aug - T+A followed by T+A (standard synonym replacement)
import json, shutil, sys, random, os, numpy as np, pickle, nltk, pdb
sys.path.insert(1, '../')
from preprocess_tools.process_utils import jsonl_save
from preprocess_tools.eda import synonym_helper
from pathlib import Path
from tqdm import tqdm

NUM_SAMPLES = 2000

res = pickle.load(open('../../stage1_processed_data/KPTimes/KPTimes_metadata.pkl', 'rb')) # since metadata file is common
vocab2idx = res['vocab2idx']

# initialize tokenizer
tokenizer = nltk.tokenize.MWETokenizer(separator='') # separator is inserted between multiple words when combining
tokenizer.add_mwe(('<','digit','>')) # strings to combine - <, digit, >
tokenizer.add_mwe(('<', 'sep', '>'))
tokenizer.add_mwe(('<', 'eos', '>'))

def process_data(file, path):
    data = []
    with open('../../stage1_processed_data/KPTimes/'+file) as f:
        for line in f:
            data.append(json.loads(line))

    if 'train' in file:
        subsample = random.sample(range(len(data)), NUM_SAMPLES)
        data = [data[idx] for idx in subsample]
    
    srcs, src_vecs, trgs, trg_vecs = [], [], [], []
    
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
            new_src     = synonym_helper(' '.join(src))
            new_src     = nltk.word_tokenize(new_src) # word_tokenize is better than simple split, splits at punctuation also
            new_src     = tokenizer.tokenize(new_src)
            new_src_vec = [vocab2idx.get(word, vocab2idx['<unk>']) for word in new_src]
            srcs.append(new_src)
            src_vecs.append(new_src_vec)
            trgs.append(data[i]['trg'])
            trg_vecs.append(data[i]['trg_vec'])
        
    data_dict = {'src': srcs, 'trg': trgs, 'src_vec': src_vecs, 'trg_vec': trg_vecs}
    jsonl_save(filepath=path+file.replace('KPTimes', 'KPTimes-lr-2000'), data_dict=data_dict)


seeds = [1433, 1896, 1922]

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    dump_folder = '../../stage2_processed_data/KPTimes-lr-2000/{}/v2_std_sr_aug/'.format(seed)
    Path(dump_folder).mkdir(parents=True, exist_ok=True)

    process_data('KPTimes_train.jsonl'   , dump_folder)
    process_data('KPTimes_dev_KPTimes.jsonl' , dump_folder)
    process_data('KPTimes_test_KPTimes.jsonl', dump_folder)
    shutil.copy('../../stage1_processed_data/KPTimes/KPTimes_metadata.pkl', dump_folder)  # Copy metadata file as it is
    os.rename(dump_folder+'KPTimes_metadata.pkl', dump_folder+'KPTimes-lr-2000_metadata.pkl')