"""
Create train/dev/test data for SEPP 2021 shared task based on Europarl
Europarl corpus: https://opus.nlpl.eu/Europarl.php
Using "leftmost column language IDs = tokenized corpus files in XML", ie.
https://opus.nlpl.eu/download.php?f=Europarl/v8/xml/en.zip
"""

__author__ = 'don.tuggener@zhaw.ch'

import io
import json
import os
import random
import string
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from typing import List, Dict
import xml.etree.ElementTree as ET

RAND_SEED = 42
random.seed(RAND_SEED)
RELEVANT_PUNCT = {'.', ',', '?', '-', ':', '!', ';'}
PUNCT_MAP = {'!': '.', ';': '.'}


def to_tsv(xml_files: List[str], zf: ZipFile, outpath: str) -> Dict:
    fname_map = dict()
    label_counts = Counter()

    for i, xml_file in enumerate(xml_files, 1):
        print('\r' + str(i) + ' ' + xml_file, end='')

        with io.TextIOWrapper(zf.open(xml_file), encoding="utf-8") as f:
            xml_str = f.read()
        root = ET.fromstring(xml_str)
        outfile = os.path.join(outpath, str(i) + '.tsv')
        fname_map[i] = xml_file

        sents = [sent for sent in root.iter(tag='s')]
        if len(sents) < 2:
            continue  # omit docs with only 1 sentence
        filt_sents = list()

        for sent in sents:

            punct_sent = False
            tokens = [tok.text.lower() for tok in sent.iter(tag='w')]
            if len(tokens) < 3:
                continue  # omit sentences with only 2 tokens (incl. punctuation)

            if tokens[-1] in string.punctuation and tokens[-1] not in RELEVANT_PUNCT:
                continue

            filt_tokens, t2_labels = list(), list()

            for j in range(len(tokens)):

                if tokens[j] in RELEVANT_PUNCT:
                    label_counts[tokens[j]] += 1
                    continue
                elif tokens[j] in string.punctuation:
                    punct_sent = True
                    break
                elif tokens[j] in {'\u200b', '\xad'}:  # Zero Width Space, soft hyphen
                    print(f'\nfound weird character {tokens[j]} in', xml_file)
                    continue
                else:
                    if tokens[j].startswith('&') and '#' in tokens[j]:  # filter some mal-formed HTML symbols, i.e. "& #160"
                        continue
                    filt_tokens.append(tokens[j])
                    if j < len(tokens) - 1:
                        if tokens[j + 1] in RELEVANT_PUNCT:  # subtask 2 label
                            t2_labels.append(PUNCT_MAP.get(tokens[j + 1], tokens[j + 1]))
                        else:
                            t2_labels.append(0)

            if len(filt_tokens) > 2 and not punct_sent:
                t1_labels = [0 for _ in filt_tokens]  # sentence end labels
                t1_labels[-1] = 1
                if len(filt_tokens) - len(t2_labels) == 1:
                    t2_labels.append('.')  # if no punctuation is detected at sentence end
                assert len(filt_tokens) == len(t1_labels) == len(t2_labels)
                filt_sents.append((filt_tokens, t1_labels, t2_labels))

        if len(filt_sents) > 1:
            with open(outfile, 'w', encoding='utf8') as f:
                for (filt_tokens, t1_labels, t2_labels) in filt_sents:
                    for tok, t1_label, t2_label in zip(filt_tokens, t1_labels, t2_labels):
                        f.write(f'{tok}\t{t1_label}\t{t2_label}\n')

    print()
    for label, cnt in label_counts.most_common():
        print(label, cnt)

    return fname_map


def create_europarl_data(europarl_zipfile: str, lang: str, outdir: str = '/tmp/sepp_nlg_2021_data/') -> None:
    Path(outdir).mkdir(exist_ok=True)
    Path(os.path.join(outdir, lang)).mkdir(exist_ok=True)
    for dir_name in {'train', 'dev', 'test'}:
        Path(os.path.join(outdir, lang, dir_name)).mkdir(exist_ok=True)
    mapping_dir = os.path.join(outdir, 'xml_files_to_tsv_files_mappings')
    Path(mapping_dir).mkdir(exist_ok=True)

    with ZipFile(europarl_zipfile, 'r') as zf:
        # The ordering of this list is not consistent across runs, will result in different splits below
        xml_files = [
            fname for fname in zf.namelist()
            if fname.startswith(f'Europarl/xml/{lang}/') and fname.endswith('.xml')
        ]
        xml_files.sort()  # NOT used for creating official data, inserted later for consistency across runs
        random.shuffle(xml_files)
        train_files, test_files = train_test_split(xml_files, test_size=.2, random_state=RAND_SEED)
        train_files, dev_files = train_test_split(train_files, test_size=.2, random_state=RAND_SEED)

        dev_fname_map = to_tsv(dev_files, zf, os.path.join(outdir, lang, 'dev'))
        with open(os.path.join(mapping_dir, f'dev_fname_map_{lang}.json'), 'w') as f:
            json.dump(dev_fname_map, f)

        test_fname_map = to_tsv(test_files, zf, os.path.join(outdir, lang, 'test'))
        with open(os.path.join(mapping_dir, f'test_fname_map_{lang}.json'), 'w') as f:
            json.dump(test_fname_map, f)

        train_fname_map = to_tsv(train_files, zf, os.path.join(outdir, lang, 'train'))
        with open(os.path.join(mapping_dir, f'train_fname_map_{lang}.json'), 'w') as f:
            json.dump(train_fname_map, f)


if __name__ == '__main__':
    create_europarl_data('data/en.zip', lang='en', outdir='data/sepp_nlg_2021_data/')
    create_europarl_data('data/de.zip', lang='de', outdir='data/sepp_nlg_2021_data/')
    create_europarl_data('data/fr.zip', lang='fr', outdir='data/sepp_nlg_2021_data/')
    create_europarl_data('data/it.zip', lang='it', outdir='data/sepp_nlg_2021_data/')