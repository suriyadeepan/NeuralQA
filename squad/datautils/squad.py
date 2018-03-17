"""
    Text Processing Nodes for
     Stanford Question Answering Dataset 
      (SQuAD)

"""
import json
import random

from pprint import pprint as pp
from nltk import word_tokenize
from tqdm import tqdm
from collections  import namedtuple
import resources as R

import sys
sys.path.append('../')

"""
    Collection of named tuples which function as containers

"""
Context = namedtuple('Context', [ 'idx', 'text', 'qas' ]) 
QA = namedtuple('QA', [ 'context_idx', 'idx', 'question', 'answers' ])
RawSample = namedtuple('RawSample', [ 'idx', 'context', 'question', 'answer' ])
Answer = namedtuple('Answer', [ 'start', 'end', 'text' ])


def read_file(filename, start=0, num_samples=100):
    """
    Read SQuAD *.json file from disk

    Args:
        filename : train or test *.json file

    Returns:
        raw text data from *.json file

    """
    with open(filename) as jfile:
        return json.load(jfile)['data']

def reduce_jsonesque_data(jdata):
    """
    Reduce JSON-esque data to (contexts, QAs)
    We sort lists at each level of hierarchy, to avoid randomness

    Args:
        jdata : raw text data in json/dict format

    Returns:
        (question-answer pairs [ <dict> ], contexts [ <text> ])

    """
    contexts = []
    qas = []
    # sort article by title
    for j, d in enumerate(sorted(jdata, key=lambda c : c['title'])): 
        # sort paragraphs by content
        for i, p in enumerate(sorted(d['paragraphs'], key=lambda p : p['context'])):
            context_idx = 'article_{}/paragraph_{}'.format(j, i)
            # fetch id's of QA-pairs
            qa_indices = [ qa['id'] for qa in p['qas'] ]
            # collect instances of Context
            contexts.append(Context(context_idx, p['context'], qa_indices))
            # collect QA-pairs
            qas.extend([ QA(context_idx, qa['id'], qa['question'], qa['answers']) 
                for qa in p['qas'] ])

    return qas, contexts

def read_squad_file(filename):
    """
    Read file (train or dev) and return dataitems

    Args:
        filename : train or test file with PATH

    Returns:
        List of partially filled DataItems

    """
    # read JSON file
    #  extract qa pairs and contexts from JSON content
    qas, contexts = reduce_jsonesque_data( read_file(filename) )
    contexts_dict = { c.idx : c for c in contexts }

    assert len(set([ qa.context_idx for qa in qas ])) == len(contexts)

    samples = []
    for i in tqdm(range(len(qas))):
        # TODO 
        #  (o) check if answer exists in context

        # build an Answer <namedtuple>
        ans_ = qas[i].answers
        if len(ans_) > 1: # if multiple answers exist (DEV)
            # choose the most agreed
            ans_ = ans_[choose_answer( [ a['text'] for a in ans_ ] )]
        else:
            ans_ = ans_[0]

        start = int(ans_['answer_start'])
        end   = start + len(ans_['text'])
        answer = Answer(start, end, ans_['text'])

        # mutate context
        context = contexts_dict[qas[i].context_idx]
        answer, context = word_index_answer(context, answer)

        samples.append(
                RawSample(qas[i].idx, # unique index
                    context,          # context  <text>
                    qas[i].question,  # question <text>
                    answer            # answer namedtuple
                    )
                )
        # check if answer exists in context
        ans, context = samples[-1].answer.text, samples[-1].context.text
        assert ans in context, (ans, '\n', context)

    return ignore_bad_samples(samples)

def choose_answer(answers):
    """
    Given a list of answers from multiple evaluators
     Choose an answer most agreed

    Args:
        answers : list of answers [ <text> ]

    Returns:
        index of best answer

    """
    def _choose_answer():
        if len(set(answers)) == len(answers):
            best_answer = max(answers, key=lambda x : len(x))
            best_answer_idx = answers.index(best_answer)
            if best_answer.isalnum():
                return best_answer_idx
            # else
            answers.remove(best_answer)
            return len(answers)-1

        return answers.index(max(set(answers), key=answers.count) )

    idx = _choose_answer()

    if len(answers[idx]) < 2 or not answers[idx].isalnum():
        answers.remove(answers[idx])
        return 0

    return idx

def __mutate_context(context, answer):
    """
    (Unused)

    """
    return Context(
            context.idx,
            context.text[:answer.start] + ' ' + answer.text + ' ' + context.text[answer.end:],
            context.qas
            )

def word_index_answer(context, answer):
    """
    Convert character indices to word indices,
     Mutate context and answer in the process

    Args:
        context : namedtuple <context>
        answer  : namedtuple <answer>

    Returns:
        mutated answer and mutated context

    """
    text = context.text
    start, end = answer.start, answer.end
    pre_text, suf_text = ' ', ' '
    pre_offset = 1
    suf_offset = 1
    mutated_text = text[:start] + pre_text + answer.text + suf_text + text[end:]

    pre = len(word_tokenize(mutated_text[:start + pre_offset]))
    ans = len(word_tokenize(mutated_text[start:end + suf_offset]))

    return Answer(pre, pre + ans, answer.text), Context(context.idx, mutated_text, context.qas)

def ignore_bad_samples(samples):
    """
    Filter out samples that don't abide

    Args:
        samples : list of samples of type (RawSample)

    Returns:
        good_samples : list of samples that abide

    """
    def is_good_sample(sample):
        t = sample.context.text
        s, e = sample.answer.start, sample.answer.end
        a = sample.answer.text
        return ' '.join(word_tokenize(t)[s:e]) == ' '.join(word_tokenize(a))

    print(':: <SQuAD> [filter]') 
    good_samples = [ sample for sample in tqdm(samples) if is_good_sample(sample) ]

    print(':: <SQuAD> [filter] {}/{} ({}%) samples ignored'.format(
        len(samples) - len(good_samples), len(samples), 
        100. * (len(samples) - len(good_samples)) / len(samples)
        ))

    return good_samples

def fetch_samples():
    """
    Wrapper around "read_squad_file"

    Args:
        None

    Returns:
        trainset, testset
        
    """
    print(':: <SQuAD> [main] Creating DEV set')
    dev   = read_squad_file(R.DEV)
    print(':: <SQuAD> [main] Creating TRAIN set')
    train = read_squad_file(R.TRAIN)
    return train, dev
