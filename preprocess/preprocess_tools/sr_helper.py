import numpy as np
import copy
import pickle
import pdb
import sys
import random
from nltk.stem import PorterStemmer
from preprocess_tools.eda import synonym_helper

# takes in tokenized_src, tokenized_trg, vocab2idx, k (number of keyphrases to be replaced with synonyms)
# returns new_src, new_src_vec, new_src_trg, new_src_vec

def keyphrase_sr_fn(src, trg, vocab2idx, k=1):
    src = copy.deepcopy(src)
    trg = copy.deepcopy(trg)
    trg = ' '.join(trg[:-1]) # remove last token (i.e., <eos>)
    kps = trg.split(' ; ')
    tokenized_kps = [kp.split(' ') for kp in kps]
    
    # collect present keyphrases based on matching b/w stemmed_kps and stemmed_src
    stemmed_kps = []
    for tokenized_kp in tokenized_kps:
        stemmed_kp = ' '.join([PorterStemmer().stem(kw.lower().strip()) for kw in copy.deepcopy(tokenized_kp)])
        stemmed_kps.append(stemmed_kp)

    stemmed_src = ' '.join([PorterStemmer().stem(token) for token in copy.deepcopy(src)])

    present_keyphrases = [] # stemmed_kps
    full_present_keyphrases = []
    for kp, stemmed_kp, tokenized_kp in zip(kps, stemmed_kps, tokenized_kps):
        if stemmed_kp in stemmed_src:
            present_keyphrases.append(stemmed_kp)
            full_present_keyphrases.append(' '.join(tokenized_kp))

    # replace k keyphrases with their synonyms
    if k == -1: # replace all keyphrases in the text
        kp_idxs_to_replace = list(range(len(present_keyphrases)))
    elif len(present_keyphrases) >= k:
        kp_idxs_to_replace = random.sample(list(range(len(present_keyphrases))), k)
    elif len(present_keyphrases) > 0:
        kp_idxs_to_replace = random.sample(list(range(len(present_keyphrases))), 1) # replace single keyphrase if k > total present keyphrases
    else:
        kp_idxs_to_replace = []        
    
    kps_replace = [1 if idx in kp_idxs_to_replace else 0 for idx in range(len(present_keyphrases))]
    # print(kps_replace)
    
    # replace (all instances of) select keyphrases in src with their synonyms
    count_unchanged = 0
    total_count = 0
    for stemmed_kp, kp, kp_replace in zip(present_keyphrases, full_present_keyphrases, kps_replace):
        tokenized_stemmed_src = [PorterStemmer().stem(token) for token in copy.deepcopy(src)]
        if kp_replace == 1:
            kp_tokens = stemmed_kp.split(' ')
            
            kp_syn        = synonym_helper(kp)
            if kp_syn == kp:
                # print(f'kp: {kp}, kp_syn: {kp_syn}')
                count_unchanged += 1
            total_count += 1
            kp_syn_tokens = kp_syn.split(' ')
            
            i = 0; new_src = []
            while i < len(src):
                if tokenized_stemmed_src[i : i+len(kp_tokens)] == kp_tokens:
                    new_src.extend(kp_syn_tokens)
                    i = i + len(kp_tokens)
                else:
                    new_src.append(src[i])
                    i += 1
            src = new_src

    src_vec = [vocab2idx.get(word, vocab2idx['<unk>']) for word in src]
    
    trg = trg.split(' ')
    trg.append('<eos>')
    trg_vec = [vocab2idx.get(word, vocab2idx['<unk>']) for word in trg]
    
    assert len(src) == len(src_vec)
    assert len(trg) == len(trg_vec)

    return src, src_vec, trg, trg_vec, total_count, count_unchanged
