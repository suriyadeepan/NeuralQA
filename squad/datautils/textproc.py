from random import shuffle
from nltk import word_tokenize, FreqDist

import resources as R
import os

import sys
sys.path.append('../')


'''
    read from file

'''
def read_file(filename):
    with open(filename, 'r') as f:
        return f.read().split('\n')[:-1]

'''
    write to file

'''
def write_file(l, filename):
    with open(filename, 'w') as f:
        for item in l:
            f.write(item)
            f.write('\n')

'''
    select column from file

'''
def select_column(filename, idx, delimiter='|'):
    result = []
    for i, l in enumerate(read_file(filename)):
        if len(l) ==  0:
            print(' length of line no.:{} is 0 line:{}'.format(i, l))
        else:
            fields = l.split(delimiter)
            result.append(fields[idx])
    return result

'''
    Flatten LoL to L

'''
def flatten(lol):
    return [ i for l in lol for i in l ]

'''
    Indices to labels (Binary)

'''
def indices_to_labels(indices, seqlen):
    return [ 1 if i in indices else 0 
            for i in range(seqlen) ]

'''
    write list of all words in sequences to disk

'''
def dump_vocabulary(sequences, frequency_threshold=0, max_vocab_size=0):
    print(':: <textproc> [dump vocabulary]')

    larg_text_chunk = ' '.join(sequences)
    words = word_tokenize(larg_text_chunk)
    # frequence distribution
    freq_dist = FreqDist(words)
    
    # words sorted by frequency
    #  most frequent on top
    words_sorted  = sorted(set(words), key=lambda w : freq_dist[w], reverse=True)

    # retain high frequency words
    vocab = [ w for w in words_sorted if freq_dist[w] >= frequency_threshold ]

    # size ceiling
    if max_vocab_size:
        vocab = vocab[:max_vocab_size]

    # check if lookup folder exists
    if not os.path.exists(R.LOOKUP):
        os.makedirs(R.LOOKUP)

    # write to file
    write_file(vocab, R.VOCAB)
    print(':: <textproc> [dump vocabulary] {} words written to lookup'.format(len(vocab)))
    return vocab

'''
    Split without bleeding

'''
def selective_split(dataitems, ratio=0.90):
    print(':: [textproc] Selective split')
    # group by indices
    groups = {}
    for item in dataitems:
        idx = item.idx
        if idx not in groups:
            groups[idx] = []
        # append to one of the groups
        groups[idx].append(item)
    # num groups
    print(':: [textproc] {} groups found'.format(
        len(groups)))
    # partition groups into 2
    indices = list(groups.keys())
    num_groups = len(indices)
    # shuffle indices
    shuffle(indices)
    partition = int(ratio*num_groups)
    # split into train/test
    train_indices = indices[ : partition]
    test_indices  = indices[partition : ]

    train = flatten([ groups[i] 
        for i in train_indices ])

    test = flatten([ groups[i]
        for i in test_indices  ])

    print(':: [textproc] {} samples split into {}% train and {}% test sets'.format(
        len(dataitems), 
        100. * len(train)/len(dataitems), 
        100. * len(test)/len(dataitems) ))

    return train, test
