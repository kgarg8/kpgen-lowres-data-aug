# v6 [random sentences] - concatenation only in training; 
# additionally instead of seqlen(T+A+Random)=800, keep full text of T+A and keep seqlen(Random)<=800

import json, shutil, random, sys, os, pdb
sys.path.insert(1, '../')
from preprocess_tools.process_utils import jsonl_save
from pathlib import Path

NUM_SAMPLES = 2000

def process_data(file, path):
    def word_count(text):
        return sum([len(element) for element in text])

    data = []
    with open('../../processed_data/LDKP10km/'+file) as f:
        for line in f:
            data.append(json.loads(line))

    if 'train' in file:
        subsample = random.sample(range(len(data)), NUM_SAMPLES)
        data = [data[idx] for idx in subsample]

    SRC_MAX_LEN = 800
    srcs, src_vecs, trgs, trg_vecs = [], [], [], []

    for i in range(len(data)):
        # recreate src, src_vec
        src     = data[i]['src']
        src_vec = data[i]['src_vec']

        indices = [i for i, x in enumerate(src) if x == '<sep>']
        indices.append(len(src))

        sentences, sentences_vec      = [], []
        for idx in range(1, len(indices) - 1):                        # start from second <sep> to leave out title & abstract
            sentences.append(src[indices[idx]+1:indices[idx+1]])      # extract sentences leaving <out> sep tokens
            sentences_vec.append(src_vec[indices[idx]+1:indices[idx+1]])

        title_abstract       = src[:indices[1]]
        title_abstract_vec   = src_vec[:indices[1]]
        src                  = title_abstract
        src_vec              = title_abstract_vec
        sep_token_id         = src_vec[indices[0]]
        total_random_src_len = sum([len(element) + 1 for element in sentences])

        if 'train' in file:
            random_src_len = 0; random_src = []; random_src_vec = []
            if total_random_src_len > SRC_MAX_LEN: # oprs
                
                if sentences != []:
                    # recreate src (list of words) of max length SRC_MAX_LEN
                    # 1) keep title & abstract as it is
                    # 2) order preserved random selection for all sentences

                    scarce_factor = SRC_MAX_LEN - random_src_len
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
                        if word_count(text) > scarce_factor:
                            rand_idxs    = random.sample(idxs, min(len(idxs), 15))
                            sublist_idxs += rand_idxs       # simply append, sort later
                            break
                    
                    sublist_idxs = sorted(sublist_idxs)
                    for idx in sublist_idxs:
                        random_src.append('<sep>')
                        random_src_vec.append(sep_token_id)
                        random_src.extend(sentences[idx])
                        random_src_vec.extend(sentences_vec[idx])
                        random_src_len += len(sentences[idx]) + 1 # +1 for <sep> token

                # ensure len(random_src) <= SRC_MAX_LEN instead of check on total seqlen
                assert len(random_src) == len(random_src_vec)
                random_src     = random_src[0:SRC_MAX_LEN]
                random_src_vec = random_src_vec[0:SRC_MAX_LEN]
                src.extend(random_src)
                src_vec.extend(random_src_vec)

            else:   # simply append sentences to title and abstract
                src          = src[:indices[1]]
                src_vec      = src_vec[:indices[1]]
                sep_token_id = src_vec[indices[0]]
                for idx in range(len(sentences)):
                    src.append('<sep>')
                    src_vec.append(sep_token_id)
                    src.extend(sentences[idx])
                    src_vec.extend(sentences_vec[idx])

        assert len(src) == len(src_vec)
        
        # append to dictionary
        srcs.append(src)
        src_vecs.append(src_vec)
        trgs.append(data[i]['trg'])
        trg_vecs.append(data[i]['trg_vec'])
        
    data_dict = {'src': srcs, 'trg': trgs, 'src_vec': src_vecs, 'trg_vec': trg_vecs}
    jsonl_save(filepath=path+file.replace('LDKP10km', 'LDKP10km-lr-2000'), data_dict=data_dict)


seeds = [1433, 1896, 1922]

for seed in seeds:
    random.seed(seed)
    dump_folder = '../../stage2_processed_data/LDKP10km-lr-2000/{}/cat_v6/'.format(seed)
    Path(dump_folder).mkdir(parents=True, exist_ok=True)

    process_data('LDKP10km_train.jsonl'   , dump_folder)
    process_data('LDKP10km_dev_LDKP10km.jsonl' , dump_folder)
    process_data('LDKP10km_test_LDKP10km.jsonl', dump_folder)
    shutil.copy('../../processed_data/LDKP10km/LDKP10km_metadata.pkl', dump_folder)  # Copy metadata file as it is
    os.rename(dump_folder+'LDKP10km_metadata.pkl', dump_folder+'LDKP10km-lr-2000_metadata.pkl')