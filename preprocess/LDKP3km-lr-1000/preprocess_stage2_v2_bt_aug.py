# lr-1000 - 1000 samples in training set
# v2_bt_aug - T+A followed by backtranslated-T+A

import json, shutil, sys, random, os, numpy as np, pickle, nltk, math, pdb
import transformers, torch, sys, pandas as pd, argparse
sys.path.insert(1, '../')
from transformers import MarianMTModel, MarianTokenizer
from preprocess_tools.process_utils import jsonl_save
from pathlib import Path
from tqdm import tqdm
transformers.logging.set_verbosity_info()

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# English to Romance languages
target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
target_model = MarianMTModel.from_pretrained(target_model_name).cuda()

# Romance languages to English
en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name).cuda()

# load vocab
res = pickle.load(open('../../stage1_processed_data/LDKP3km/LDKP3km_metadata.pkl', 'rb')) # since metadata file is common
vocab2idx = res['vocab2idx']

# initialize tokenizer
tokenizer = nltk.tokenize.MWETokenizer(separator='') # separator is inserted between multiple words when combining
tokenizer.add_mwe(('<','digit','>')) # strings to combine - <, digit, >
tokenizer.add_mwe(('<', 'sep', '>'))
tokenizer.add_mwe(('<', 'eos', '>'))


def translate(texts, model, lang_tokenizer, language="fr", num_beams=1):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    encoded = lang_tokenizer.prepare_seq2seq_batch(src_texts, return_tensors='pt').to(device)
    
    # Generate translation using model
    if num_beams > 1:
        translated = model.generate(**encoded, do_sample=False, top_k=0, num_beams=num_beams)
    else:
        translated = model.generate(**encoded, do_sample=True, max_length=512, top_k=0, num_beams=1, temperature=0.7)

    # Convert the generated tokens indices back into text
    translated_texts = lang_tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return translated_texts


def back_translate(texts, source_lang="en", target_lang="fr", num_beams=1):
    # Translate from source to target language
    fr_texts = translate(texts, target_model, target_tokenizer, language=target_lang, num_beams=num_beams)

    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer, language=source_lang, num_beams=num_beams)
    
    return back_translated_texts


def process_data(file, path):
    data = []
    with open('../../stage1_processed_data/LDKP3km/'+file) as f:
        for line in f:
            data.append(json.loads(line))
    
    NUM_SAMPLES = 1000
    if 'train' in file:
        subsample = random.sample(range(len(data)), NUM_SAMPLES)
        data = [data[idx] for idx in subsample]

    srcs, src_vecs, trgs, trg_vecs = [], [], [], []
    
    for i in tqdm(range(len(data))):
        indices = [i for i, x in enumerate(data[i]['src']) if x == '<sep>']
        src     = data[i]['src'][:indices[1]]
        src_vec = data[i]['src_vec'][:indices[1]]

        # append original T+A to dictionary
        srcs.append(src)
        src_vecs.append(src_vec)
        trgs.append(data[i]['trg'])
        trg_vecs.append(data[i]['trg_vec'])

        if 'train' in file:
            src_segments = []
            title        = data[i]['src'][:indices[0]]
            abstract     = data[i]['src'][indices[0]+1:indices[1]]
            
            # segment T+A into blocks of 25 words. We restrict to 25 words, o/w `max_length` ends up truncating some text
            # 25 words is just a heuristic to create blocks. we could have also segmented acc to sentences.

            # add Title as first segment
            src_segments.append(' '.join(title))
            
            total = 0
            while (total < len(abstract)):
                src_segments.append(' '.join(abstract[total: total+25]))
                total += 25

            aug_text = back_translate(src_segments, source_lang="en", target_lang="fr", num_beams=1)
            
            # combine aug_text segments into new_src
            aug_text_title    = aug_text[0]
            aug_text_abstract = aug_text[1:]
            aug_text_abstract = ' '.join(aug_text_abstract)
            aug_text_string   = aug_text_title + ' <sep> ' + aug_text_abstract
            aug_text_string   = aug_text_string.lower()
            new_src           = nltk.word_tokenize(aug_text_string) # word_tokenize is better than simple split, splits at punctuation also
            new_src           = tokenizer.tokenize(new_src)
            new_src_vec       = [vocab2idx.get(word, vocab2idx['<unk>']) for word in aug_text]
            
            # append backtranslated T+A to dictionary
            srcs.append(new_src)
            src_vecs.append(new_src_vec)
            trgs.append(data[i]['trg'])
            trg_vecs.append(data[i]['trg_vec'])

            # display
            if i % 500 == 0:
                print('Original sample:\n', ' '.join(src))
                print('backtranslated sample:\n', ' '.join(new_src))
        
    data_dict = {'src': srcs, 'trg': trgs, 'src_vec': src_vecs, 'trg_vec': trg_vecs}
    jsonl_save(filepath=path+file.replace('LDKP3km', 'LDKP3km-lr-1000'), data_dict=data_dict)


seeds = [1433, 1896, 1922]

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    dump_folder = '../../stage2_processed_data/LDKP3km-lr-1000/{}/v2_bt_aug/'.format(seed)
    Path(dump_folder).mkdir(parents=True, exist_ok=True)

    process_data('LDKP3km_train.jsonl'   , dump_folder)
    process_data('LDKP3km_dev_LDKP3km.jsonl' , dump_folder)
    process_data('LDKP3km_test_LDKP3km.jsonl', dump_folder)
    shutil.copy('../../stage1_processed_data/LDKP3km/LDKP3km_metadata.pkl', dump_folder)  # Copy metadata file as it is
    os.rename(dump_folder+'LDKP3km_metadata.pkl', dump_folder+'LDKP3km-lr-1000_metadata.pkl')