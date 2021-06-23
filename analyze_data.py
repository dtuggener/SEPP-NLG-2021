"""
Analytics for SEPP-NLG 2021: https://sites.google.com/view/sentence-segmentation/
"""

__author__ = 'don.tuggener@zhaw.ch'

import io
import os
from zipfile import ZipFile
from typing import List, Tuple
from collections import Counter


def get_tokens_and_sentences(data_zip: str, lang: str, data_set: str) -> Tuple[List[str], List[str]]:
    with ZipFile(data_zip, 'r') as zf:
        relevant_dir = os.path.join('sepp_nlg_2021_data', lang, data_set)
        tsv_files = [
            fname for fname in zf.namelist()
            if fname.startswith(relevant_dir) and fname.endswith('.tsv')
        ]
        tokens, sentences = list(), list()
        for i, tsv_file in enumerate(tsv_files, 1):
            print('\r' + str(i), tsv_file, end='')
            with io.TextIOWrapper(zf.open(tsv_file), encoding="utf-8") as f:
                tsv_str = f.read()
            lines = tsv_str.strip().split('\n')
            toks = list()
            for line in lines:
                cells = line.split('\t')
                toks.append(cells[0])
                if cells[1] == '1':
                    sentences.append( ' '.join(toks))
                    tokens.extend(toks)
                    toks = list()
        print()
    return tokens, sentences


def train_test_dev_overlap(data_zip: str, lang: str) -> None:

    print('analysing', lang)
    data_sets = {'train', 'dev', 'test'}

    tokens_per_ds, token_set_per_ds, sents_per_ds, sents_set_per_ds = dict(), dict(), dict(), dict()
    shared_sent_counts_uniq, shared_sent_counts = Counter(), Counter()

    for ds in data_sets:
        print('loading', ds)
        tokens, sentences = get_tokens_and_sentences(data_zip, lang, ds)
        tokens_per_ds[ds] = tokens
        token_set_per_ds[ds] = set(tokens)
        sents_per_ds[ds] = sentences
        sents_set_per_ds[ds] = set(sentences)

    print('STATISTICS')
    print('Number of (unique) sentences:')
    for ds in data_sets:
        print(f'{ds} {len(sents_per_ds[ds])} {len(sents_set_per_ds[ds])}')

    print('Number of (unique) tokens:')
    for ds in data_sets:
        print(f'{ds} {len(tokens_per_ds[ds])} {len(token_set_per_ds[ds])}')

    print('OVERLAPS')
    for sent in sents_set_per_ds['train']:
        if sent in sents_set_per_ds['dev']:
            if sent in sents_set_per_ds['test']:
                shared_sent_counts_uniq['train_dev_test'] += 1
                shared_sent_counts['train_dev_test'] += sents_per_ds['test'].count(sent)
            else:
                shared_sent_counts_uniq['train_dev'] += 1
                shared_sent_counts['train_dev'] += sents_per_ds['dev'].count(sent)
        elif sent in sents_set_per_ds['test']:
            shared_sent_counts_uniq['train_test'] += 1
            shared_sent_counts['train_test'] += sents_per_ds['test'].count(sent)

    for sent in sents_set_per_ds['dev']:
        if sent in sents_set_per_ds['test']:
            shared_sent_counts_uniq['dev_test'] += 1
            shared_sent_counts['dev_test'] += sents_per_ds['dev'].count(sent)

    print('Number of overlapping sentences (unique):')
    for k, v in shared_sent_counts.most_common():
        print(k, v, shared_sent_counts_uniq[k])

    print('Vocabulary overlap:')
    print('train <-> dev', len(token_set_per_ds['train'].intersection(token_set_per_ds['dev'])))
    print('train <-> test', len(token_set_per_ds['train'].intersection(token_set_per_ds['test'])))
    print('test <-> dev', len(token_set_per_ds['test'].intersection(token_set_per_ds['dev'])))


if __name__ == '__main__':
    data_zip = 'data/sepp_nlg_2021_data_v5.zip'
    for lang in ['en', 'de', 'fr', 'it']:
        train_test_dev_overlap(data_zip, lang)