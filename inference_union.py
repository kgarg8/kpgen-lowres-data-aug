# load the checkpoint of two models separately
# load common test dataset, vocab
# combine the predictions from the two models
# do the evaluation

import pickle, torch as T, torch.nn as nn, nltk, fasttext, numpy as np, pdb, time, argparse
from pathlib import Path
from controllers.attribute_controller import prepare_attributes
from utils.data_utils import load_dataset
from parser import get_args
from configs.configLoader import load_config
from models import *
from collaters import *
from agents import *
from torch.utils.data import Dataset, DataLoader
from utils.evaluation_utils import evaluate
from controllers.metric_controller import metric_fn
from tqdm import tqdm
from scipy import spatial


idx2vocab = {}; vocab = {}

def load_data(test_path, metadata, args):
    global vocab, idx2vocab
    data = {}
    data['vocab2idx'] = metadata['vocab2idx']
    vocab2idx         = data['vocab2idx']
    data['PAD_id']    = vocab2idx['<pad>']
    data['UNK_id']    = vocab2idx['<unk>'] if '<unk>' in vocab2idx else None
    data['SEP_id']    = vocab2idx['<sep>'] if '<sep>' in vocab2idx else None
    data['vocab_len'] = len(vocab2idx)
    data['idx2vocab'] = {id: token for token, id in vocab2idx.items()}
    data['test']      = load_dataset(test_path, limit=args.limit)
    idx2vocab         = data['idx2vocab']
    vocab             = data['vocab2idx']
    return data


def load_infer_checkpoint(args, run_time=0):

    metadata_path = Path('processed_data/{}/{}/{}_metadata.pkl'.format(args.lr_seed, args.version, args.dataset))
    with open(metadata_path, 'rb') as fp:
        metadata = pickle.load(fp)

    ckpt_path = Path('experiments/{}/{}/saved_weights/{}_{}_{}/{}.pt'.format(args.lr_seed, args.version, args.dataset, args.model, args.model_type, run_time))
    print(ckpt_path)
    checkpoint = T.load(ckpt_path)
    
    test_path  = Path('processed_data/{}/{}/{}_test_{}.jsonl'.format(args.lr_seed, args.version, args.dataset, args.dataset))
    data       = load_data(test_path, metadata, args)
    attributes = prepare_attributes(data, args)

    model = eval("{}_model".format(args.model_type))
    model = model(attributes=attributes, config=config)
    model = model.to(args.device)

    if config['DataParallel']:
        model = nn.DataParallel(model)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # since we do only inference here

    collater = eval('{}_collater'.format(args.model))
    dev_collater = collater(PAD=data['PAD_id'], config=config, train=False)

    dataloader = DataLoader(data['test'], batch_size=config['dev_batch_size'], num_workers=config['num_workers'], shuffle=False, collate_fn=dev_collater.collate_fn)

    return model, dataloader
    

def decode(prediction_idx, src):
    decoded_prediction = []
    for id in prediction_idx:
        if id >= len(vocab):
            decoded_prediction.append(src[id - len(vocab)])
        else:
            decoded_prediction.append(idx2vocab[id])
    return ' '.join(decoded_prediction)


def run(model, batch):
    with T.no_grad():
        output_dict = model(batch)
    
    logits      = output_dict['logits']
    predictions = T.argmax(logits, dim=-1)
    predictions = predictions.cpu().detach().numpy().tolist()    
    predictions = [decode(prediction, src) for prediction, src in zip(predictions, batch['src'])]

    return predictions


def getlist_from_trg(trg):
    candidate_list = []
    trg = trg.split(';')[:-1]
    for kp in trg: # get rid of last candidate <eos>
        kp = kp.strip()
        candidate_list.append(kp)
    candidate_list = list(set(candidate_list))
    return candidate_list


def cosine_distance_wordembedding_method(word2vec_model, s1, s2):                                                                                             
    vec1   = np.mean([word2vec_model[word] for word in s1],axis=0)                                                                                            
    vec2   = np.mean([word2vec_model[word] for word in s2],axis=0)                                                                                            
    if not np.any(vec1) or not np.any(vec2):                                                                                                                  
        print('Zero embeddings')                                                                                                                              
    cosine = spatial.distance.cosine(vec1, vec2)                                                                                                              
    return round((1-cosine)*100, 2)  


def semantic_candidates(stemmer, tokenizer, fasttext_model, input, candidates, TOP_K):
    tokenized_input = nltk.word_tokenize(input)
    tokenized_input = tokenizer.tokenize(tokenized_input)  # merge tokens with multi-words
    stemmed_input   = [stemmer.stem(token) for token in tokenized_input]
    
    distances = []
    for cand in candidates:
        tokenized_cand = nltk.word_tokenize(cand)
        tokenized_cand = tokenizer.tokenize(tokenized_cand) # merge tokens with multi-words
        stemmed_cand   = [stemmer.stem(token) for token in tokenized_cand]
        distances.append(cosine_distance_wordembedding_method(fasttext_model, stemmed_input, stemmed_cand))
    
    if TOP_K == -1:
        TOP_K = len(candidates)

    distances        = np.array(distances)
    t1               = np.argsort(-distances)[:TOP_K]
    top_cand_idx     = np.argsort(-distances)[:TOP_K] # -ve distances because argsort sorts in increasing order
    distances_sorted = distances[top_cand_idx]
    final_candidates = [candidates[idx] for idx in top_cand_idx]

    return distances_sorted, final_candidates


def post_process(stemmer, tokenizer, fasttext_model, predictions, predictions2, inputs, ranked_union_flag=False):
    combined_preds = []

    for input, pred, pred2 in zip(inputs, predictions, predictions2):
        p1 = pred.split(' <eos>')[0]
        p2 = pred2.split(' <eos>')[0]
        union = p1 + ' ; ' + p2 + ' <eos>'
        # print('Simple Union: \n', union)
        candidate_list = getlist_from_trg(union)
        
        if ranked_union_flag: # rank using fasttext model
            _, ranked_union = semantic_candidates(stemmer, tokenizer, fasttext_model, ' '.join(input), candidate_list, TOP_K=-1)
        else: # no need to rank
            ranked_union = candidate_list
        
        ranked_union = ' ; '.join(ranked_union)
        ranked_union = ranked_union.strip(' ; ')
        ranked_union += ' <eos>'
        # print('Ranked union: \n', ranked_union)
        combined_preds.append(ranked_union)
    return combined_preds

if __name__ == '__main__':

    parser = get_args()
    parser2 = argparse.ArgumentParser(add_help=False, parents=[parser])
    parser2.add_argument('--version2', type=str, default='')
    parser2.add_argument('--ranked_union_flag', action='store_true')
    args = parser2.parse_args()
    print(args)
    print('Ranked union flag: ', args.ranked_union_flag)

    config = load_config(args)
    config['generate'] = True
    config['cite_sep'] = False
    args.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    stemmer   = nltk.PorterStemmer()
    tokenizer = nltk.tokenize.MWETokenizer(separator='') # separator is inserted between multiple words when combining
    tokenizer.add_mwe(('<','digit','>')) # strings to combine - <, digit, >
    tokenizer.add_mwe(('<', 'sep', '>'))
    tokenizer.add_mwe(('<', 'eos', '>'))
    fasttext_model = fasttext.load_model('../fastText/cc.en.300.bin')

    for lr_seed in [1433, 1896, 1922]:
        args.lr_seed = lr_seed # override lr_seed
        for run_time in [0, 1, 2]:
            # args.version = 'aug_v6' # override version
            model1, dbloader = load_infer_checkpoint(args, run_time)

            args2 = copy.deepcopy(args)
            args2.version = args.version2
            print('Doing union over {} & {}'.format(args.version, args2.version))
            model2, _ = load_infer_checkpoint(args2, run_time) # since vocab & test_data for both models is same if root version (i.e., v6) is same
            
            metrics_list = []
            
            for batches in tqdm(dbloader):
                for batch_id, batch in enumerate(batches):
                    # get predictions from both the models and play around with it
                    predictions  = run(model1, batch)
                    predictions2 = run(model2, batch)
                    union = post_process(stemmer, tokenizer, fasttext_model, predictions, predictions2, batch['src'], args.ranked_union_flag)
                    metrics = evaluate(copy.deepcopy(batch['src']), copy.deepcopy(batch['trg']), copy.deepcopy(union))
                    metrics['loss'] = 0.0 # don't care thing
                    metrics_list.append(metrics)
            
            test_metric = metric_fn(metrics_list, args)

            display_string = ""
            for k, v in test_metric.items():
                display_string += "{}: {}; ".format(k, v)
            display_string += "\n"
            print(display_string)

            folder = Path('results/{}/union_{}__{}/{}_{}_{}/'.format(args.lr_seed, args.version, args2.version, args.dataset, args.model, args.model_type))
            folder.mkdir(parents=True, exist_ok=True)
            f = open('results/{}/union_{}__{}/{}_{}_{}/{}.txt'.format(args.lr_seed, args.version, args2.version, args.dataset, args.model, args.model_type, run_time), 'w')
            f.write(display_string)