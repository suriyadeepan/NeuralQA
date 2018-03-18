from datautils.squad import fetch_samples
from datautils.textproc import dump_vocabulary
import os
import pickle

import resources as R
from nltk import word_tokenize


def cache_data(trainset, testset):
    print(':: <data> Caching preprocessed data')
    if not os.path.exists(R.CACHE):
        os.makedirs(R.CACHE)
        
    with open(R.CACHE + '/train.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open(R.CACHE + '/test.pkl', 'wb') as f:
        pickle.dump(testset, f)

def load_cache():
    try:
        with open(R.CACHE + '/train.pkl', 'rb') as f:
            trainset = pickle.load(f)
        with open(R.CACHE + '/test.pkl', 'rb') as f:
            testset = pickle.load(f)
        return trainset, testset
    except:
        return None

def fetch_squad_data(_sort=True, _flush=False):

    if not _flush and load_cache():
        print(':: flush -> False,  loading from disk')
        return load_cache()

    else:
        trainset, testset = fetch_samples()

        # sort
        if _sort:
            trainset = sorted(trainset, 
                    key=lambda x : len(word_tokenize(x.context.idx))
                    )
            testset  = sorted(testset, 
                    key=lambda x : len(word_tokenize(x.context.idx))
                    )

        # dump vocabulary to R.VOCAB
        dump_vocabulary(  # combine text from context and question
                [ x.context.text + ' ' + x.question for x in testset + trainset ],
                frequency_threshold = R.VOCAB_FREQ, 
                max_vocab_size = R.VOCAB_MAX_SIZE
                )

        # cache data
        cache_data(trainset, testset)

        return trainset, testset


if __name__ == '__main__':

    train, test = fetch_squad_data(_flush=True)
