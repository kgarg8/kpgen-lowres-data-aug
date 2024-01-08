import numpy as np
import copy
import pickle
import pdb
from nltk.stem import PorterStemmer


# takes in tokenized_src, tokenized_trg, vocab2idx, prob value with which keyphrases should be masked
# returns new_src, new_src_vec, new_src_trg, new_src_vec

def keyphrase_dropout_fn(src, trg, vocab2idx, delete=False, p=0.7):
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
    for kp, stemmed_kp in zip(kps, stemmed_kps):
        if stemmed_kp in stemmed_src:
            present_keyphrases.append(stemmed_kp)

    # drop keyphrases with probability p
    kps_drop = np.random.binomial(1, p, len(present_keyphrases))

    # replace the dropped keyphrases in src with '#'
    tokenized_stemmed_src = [PorterStemmer().stem(token) for token in copy.deepcopy(src)]
    for kp, kp_drop in zip(present_keyphrases, kps_drop):
        if kp_drop == 1:
            kp_tokens = kp.split(' ')
            for i in range(len(src)):
                if tokenized_stemmed_src[i : i+len(kp_tokens)] == kp_tokens:
                    src[i : i+len(kp_tokens)] = ['#' for token in kp_tokens]
    
    if delete: # delete masked tokens
        src = [token for token in src if token != '#']

    src_vec = [vocab2idx.get(word, vocab2idx['<unk>']) for word in src]
    
    trg = trg.split(' ')
    trg.append('<eos>')
    trg_vec = [vocab2idx.get(word, vocab2idx['<unk>']) for word in trg]
    
    assert len(src) == len(src_vec)
    assert len(trg) == len(trg_vec)

    return src, src_vec, trg, trg_vec
