"""
Create surprise test set for SEPP-NLG 2021
TED talk corpus: https://opus.nlpl.eu/TED2020.php
i.e.: https://opus.nlpl.eu/download.php?f=TED2020/v1/xml/en.zip
Find top N talks that diverge the most from the vocabulary in the Europarl data
"""

import io
import json
import os
import string
import xml.etree.ElementTree as ET
from typing import Dict, Optional
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
from analyze_data import get_tokens_and_sentences
from create_shared_task_data import RELEVANT_PUNCT, PUNCT_MAP


def get_testset_vocab(data_zip: str, lang: str, data_set: str) -> Counter:
    tokens, _ = get_tokens_and_sentences(data_zip, lang, data_set)
    return Counter(tokens)


def sample_surprise_files(data_zip: str, lang: str, testset_vocab: Counter, num_files: Optional[int] = 500,
                          outdir: Optional[str] = 'sepp_nlg_2021_surprise_testset/') -> Dict:
    Path(outdir).mkdir(exist_ok=True)
    Path(os.path.join(outdir, lang)).mkdir(exist_ok=True)
    Path(os.path.join(outdir, lang, 'surprise_test')).mkdir(exist_ok=True)
    Path(os.path.join(outdir, lang, 'surprise_train')).mkdir(exist_ok=True)
    outdir_train = os.path.join(outdir, lang, 'surprise_train')
    outdir = os.path.join(outdir, lang, 'surprise_test')
    fname_map = dict()
    sorted_files = list()
    testset_vocab_set = set(testset_vocab.keys())

    with ZipFile(data_zip, 'r') as zf:
        xml_files = [
            fname for fname in zf.namelist()
            if  fname.endswith('.xml')
        ]

        for i, xml_file in enumerate(xml_files):
            print('\r' + str(i) + ' ' + xml_file, end='')
            with io.TextIOWrapper(zf.open(xml_file), encoding="utf-8") as f:
                xml_str = f.read()
            root = ET.fromstring(xml_str)

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

                # Filter (laughter) and (applause) etc; replace '--' with '-'
                while tokens[0] == '(' and tokens[2] == ')':
                    tokens = tokens[3:]
                tokens = ['-' if tok == '--' else tok for tok in tokens]
                import unicodedata
                for j in range(len(tokens)):

                    if tokens[j] in RELEVANT_PUNCT:
                        continue
                    elif tokens[j] in string.punctuation:
                        punct_sent = True
                        break
                    elif len(tokens[j]) == 1 and unicodedata.category(tokens[j]) in {'Cc', 'Cf'}:
                        print(f'\nfound weird character {tokens[j]} in', xml_file)
                        punct_sent = True
                        break
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

            if len(filt_sents) >= 50:  # shorter ones are songs etc. sometimes
                vocab = list()
                for sent in filt_sents:
                    vocab.extend(sent[0])
                vocab = Counter(vocab)
                vocab_set = set(vocab.keys())
                overlap = len(vocab_set.intersection(testset_vocab_set)) / len(vocab_set)
                sorted_files.append((overlap, xml_file, filt_sents, vocab))

    print()
    sorted_files.sort()  # sort by % of vocab overlap
    overlap_percentages = list()
    for i, data in enumerate(sorted_files[:num_files]):
        outfile = os.path.join(outdir, str(i) + '.tsv')
        fname_map[i] = data[1]
        overlap_percentages.append(data[0])
        with open(outfile, 'w', encoding='utf8') as f:
            for toks, t1_labels, t2_labels in data[2]:
                for tok, t1_label, t2_label in zip(toks, t1_labels, t2_labels):
                    f.write(f'{tok}\t{t1_label}\t{t2_label}\n')
    for i, data in enumerate(sorted_files[num_files:]):
        outfile = os.path.join(outdir_train, str(i) + '.tsv')
        # fname_map[i] = data[1]
        with open(outfile, 'w', encoding='utf8') as f:
            for toks, t1_labels, t2_labels in data[2]:
                for tok, t1_label, t2_label in zip(toks, t1_labels, t2_labels):
                    f.write(f'{tok}\t{t1_label}\t{t2_label}\n')

    print(f'avg overlap %: {sum(overlap_percentages)/len(overlap_percentages)}')
    return fname_map


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create surprise test set for SEPP-NLG 2021')
    parser.add_argument("testset_data_zip", help="path to data zip file, e.g. 'data/sepp_nlg_2021_v5.zip'")
    parser.add_argument("ted_data_zip", help="path to data zip containing the TED talk XMLs, e.g. 'data/en.zip'")
    parser.add_argument("language", help="target language ('en', 'de', 'fr', 'it', ; i.e. one (or all) of the subfolders in the zip file's main folder)")
    parser.add_argument("outdir", help="folder to store data (language and dataset subfolders will be created automatically)",  nargs="?", default='data/sepp_nlg_2021_surprise_testset/')
    args = parser.parse_args()

    data_set = 'test'
    mapping_dir = 'xml_files_to_tsv_files_mappings'
    testset_vocab = get_testset_vocab(args.testset_data_zip, args.language, data_set)
    fname_map = sample_surprise_files(args.ted_data_zip, args.language, testset_vocab, outdir=args.outdir)
    with open(os.path.join(mapping_dir, f'surprise_test_fname_map_{args.language}.json'), 'w') as f:
        json.dump(fname_map, f)
