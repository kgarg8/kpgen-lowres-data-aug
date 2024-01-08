# Augmentation for v4
# Train: article1 = T+A; article2 = random sentences[OPRS][seqlen=800]

import json, shutil, sys, random, os, pdb
sys.path.insert(1, '../')
from tqdm import tqdm
from preprocess_tools.process_utils import jsonl_save
from pathlib import Path

NUM_SAMPLES = 2000

def process_data(file, path):
    def word_count(text):
        return sum([len(element) for element in text])

    data = []
    with open('../../stage1_processed_data/KPTimes/'+file) as f:
        for line in f:
            data.append(json.loads(line))
    
    if 'train' in file:
        subsample = random.sample(range(len(data)), NUM_SAMPLES)
        data = [data[idx] for idx in subsample]

    SRC_MAX_LEN = 800
    srcs, src_vecs, trgs, trg_vecs = [], [], [], []

    for i in tqdm(range(len(data))):
        # recreate src, src_vec
        src     = data[i]['src']
        src_vec = data[i]['src_vec']

        indices = [i for i, x in enumerate(src) if x == '<sep>']
        indices.append(len(src))

        sentences, sentences_vec      = [], []
        for idx in range(1, len(indices) - 1):                        # start from second <sep> to leave out title & abstract
            sentences.append(src[indices[idx]+1:indices[idx+1]])      # extract sentences leaving <out> sep tokens
            sentences_vec.append(src_vec[indices[idx]+1:indices[idx+1]])

        sep_token_id       = src_vec[indices[0]]

        # article1 will be just T+A
        title_abstract     = src[:indices[1]]
        title_abstract_vec = src_vec[:indices[1]]
        srcs.append(title_abstract)
        src_vecs.append(title_abstract_vec)
        trgs.append(data[i]['trg'])
        trg_vecs.append(data[i]['trg_vec'])

        if 'train' in file:
            # article2 will be random sentences
            src     = []
            src_vec = []
            src_len = sum([len(element) + 1 for element in sentences])

            if src_len > SRC_MAX_LEN: # oprs
                
                if sentences != []:
                    # recreate src (list of words) of max length SRC_MAX_LEN
                    # order preserved random selection for all sentences

                    scarce_factor = SRC_MAX_LEN - len(src)
                    sublist_idxs  = []
                    idxs          = [*range(len(sentences))]
                    counter       = 0
                    while(1):
                        counter += 1
                        if counter >= 8:    break
                        rand_idxs    = random.sample(idxs, min(len(idxs), 15))
                        sublist_idxs += rand_idxs           # simply append, sort later
                        idxs         = list(set(idxs) - set(rand_idxs))
                        text         = [sentences[i] for i in sublist_idxs]
                        if word_count(text) + len(src) > scarce_factor:
                            rand_idxs    = random.sample(idxs, min(len(idxs), 15))
                            sublist_idxs += rand_idxs       # simply append, sort later
                            break
                    
                    sublist_idxs = sorted(sublist_idxs)
                    for idx in sublist_idxs:
                        src.extend(sentences[idx])
                        src_vec.extend(sentences_vec[idx])
                        src.append('<sep>')
                        src_vec.append(sep_token_id)

            else:   # simply append sentences to title and abstract
                for idx in range(len(sentences)):
                    src.extend(sentences[idx])
                    src_vec.extend(sentences_vec[idx])
                    src.append('<sep>')
                    src_vec.append(sep_token_id)

            assert len(src) == len(src_vec)
            src     = src[0:SRC_MAX_LEN]
            src_vec = src_vec[0:SRC_MAX_LEN]
            
            # append to dictionary
            srcs.append(src)
            src_vecs.append(src_vec)
            trgs.append(data[i]['trg'])
            trg_vecs.append(data[i]['trg_vec'])
        
    data_dict = {'src': srcs, 'trg': trgs, 'src_vec': src_vecs, 'trg_vec': trg_vecs}
    jsonl_save(filepath=path+file.replace('KPTimes', 'KPTimes-lr-2000'), data_dict=data_dict)

seeds = [1433, 1896, 1922]

for seed in seeds:
    random.seed(seed)
    dump_folder = '../../stage2_processed_data/KPTimes-lr-2000/{}/aug_v6/'.format(seed)
    Path(dump_folder).mkdir(parents=True, exist_ok=True)

    process_data('KPTimes_train.jsonl'   , dump_folder)
    process_data('KPTimes_dev_KPTimes.jsonl' , dump_folder)
    process_data('KPTimes_test_KPTimes.jsonl', dump_folder)
    shutil.copy('../../stage1_processed_data/KPTimes/KPTimes_metadata.pkl', dump_folder)  # Copy metadata file as it is
    os.rename(dump_folder+'KPTimes_metadata.pkl', dump_folder+'KPTimes-lr-2000_metadata.pkl')