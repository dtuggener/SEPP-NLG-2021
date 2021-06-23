"""
Bert-based baseline for punctuation prediction
"""

__author__ = 'don.tuggener@zhaw.ch'

import io
import os
import random
from pathlib import Path
from typing import Optional
from zipfile import ZipFile
from simpletransformers.ner import NERModel, NERArgs
random.seed(42)

MODEL_TYPE, MODEL_NAME = "bert", "bert-base-multilingual-uncased"
MAX_SEQ_LEN = 512  # depends on selected pretrained model
LABEL_LIST = ['.', ',', '?', '-', ':', '0']


def data_to_disk(data_zip: str, lang_code: str, data_set: str, max_len: Optional[int] = 512) -> None:
    tok_count, curr_sent_id = 0, 0

    with ZipFile(data_zip, 'r') as zf:
        if lang_code == 'all':
            langs = ['en', 'fr', 'de', 'it']
        else:
            langs = [lang_code]
        tsv_files = list()
        for lang in langs:
            relevant_dir = os.path.join('sepp_nlg_2021_data', lang, data_set)
            tsv_files.extend([
                fname for fname in zf.namelist()
                if fname.startswith(relevant_dir) and fname.endswith('.tsv')
            ])
        random.shuffle(tsv_files)

        with open(f'data/{lang_code}_{data_set}.txt', 'w', encoding='utf8') as outfile:
            for i, tsv_file in enumerate(tsv_files, 1):
                print('\r' + str(i) + ' ' + tsv_file, end='')
                with io.TextIOWrapper(zf.open(tsv_file), encoding="utf-8") as f:
                    tsv_str = f.read()
                lines = tsv_str.strip().split('\n')
                for line in lines:
                    if tok_count == max_len:
                        outfile.write('\n')
                        tok_count = 0
                    line = line.split('\t')
                    outfile.write(f'{line[0]} {line[2]}\n')
                    tok_count += 1


def train(data_zip: str, lang: str, data_set: str, max_seq_len: int, rewrite_files: bool = False,
          lazy_loading: bool = True) -> None:
    if rewrite_files:
        data_to_disk(data_zip, lang, data_set, max_len=max_seq_len)
        data_to_disk(data_zip, lang, 'train', max_len=max_seq_len)
    eval_data = f'data/{lang}_{data_set}.txt'
    train_data = f'data/{lang}_train.txt'

    model_args = NERArgs()
    model_args.labels_list = LABEL_LIST
    model_args.train_batch_size = 12
    model_args.evaluate_during_training = True
    model_args.do_lower_case = True
    model_args.num_train_epochs = 5
    model_args.overwrite_output_dir = True
    model_args.use_cuda = True
    model_args.max_seq_length = MAX_SEQ_LEN

    if lazy_loading:
        model_args.lazy_loading = True

    model = NERModel(MODEL_TYPE, MODEL_NAME, args=model_args)
    model.train_model(train_data, eval_data=eval_data)


def predict_punct(model_dir: str, data_zip: str, lang: str, data_set: str, max_len: int, outdir: Optional[str] = '/tmp/') -> None:

    outdir = os.path.join(outdir, lang, data_set)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(f'Loading model {model_dir}')
    model = NERModel(MODEL_TYPE, model_dir)
    model.args.max_seq_length = MAX_SEQ_LEN

    with ZipFile(data_zip, 'r') as zf:
        fnames = zf.namelist()
        relevant_dir = os.path.join('sepp_nlg_2021_data', lang, data_set)
        tsv_files = [
            fname for fname in fnames
            if fname.startswith(relevant_dir) and fname.endswith('.tsv')
        ]

        for tsv_file in tsv_files:
            print(tsv_file)
            with io.TextIOWrapper(zf.open(tsv_file), encoding="utf-8") as f:
                tsv_str = f.read()
            tsv_lines = tsv_str.strip().split('\n')
            lines = [line.strip().split('\t') for line in tsv_lines]
            tokens = [line[0] for line in lines]

            batches = [tokens[split_ix: split_ix+max_len]  # segment token list into lists of max_len
                       for split_ix in range(0, len(tokens), max_len)]
            preds, _ = model.predict(batches, split_on_space=False)
            all_preds = [tok for pred in preds for tok in pred]

            if not len(all_preds) == len(tokens):
                # lower max seq len until predictions are now longer truncated
                # necessary when there are lots of foreign language / unknown tokens
                # and the list of subword tokens gets too long (i.e. > MAX_SEQ_LEN)
                new_max_len = max_len
                while not len(all_preds) == len(tokens) and new_max_len > 2:
                    new_max_len = int(new_max_len * .75)
                    batches = [tokens[split_ix: split_ix + new_max_len]
                               for split_ix in range(0, len(tokens), new_max_len)]
                    preds, _ = model.predict(batches, split_on_space=False)
                    all_preds = [tok for pred in preds for tok in pred]

            if not len(all_preds) == len(tokens):
                print('miss-match prediction length')
                breakpoint()

            with open(os.path.join(outdir, os.path.basename(tsv_file)), 'w',
                      encoding='utf8') as of:
                for tok in all_preds:
                    tok_str, label_st2 = list(tok.keys())[0], list(tok.values())[0]
                    label_st1 = 1 if label_st2 in {'.', '?'} else 0
                    of.write(f'{tok_str}\t{label_st1}\t{label_st2}\n')


def main(data_zip: str, lang: str, data_set: str, outdir: str, mode: str, rewrite_files: Optional[bool] = False,
         lazy_loading: Optional[bool] = True, model_dir: Optional[str] = None) -> None:
    assert mode in {'train', 'test'}, f"Mode {mode} not supported, only 'train' and 'test'"
    max_seq_len = int(.75 * MAX_SEQ_LEN)  # leave some of max len for subword tokenization
    if mode == 'train':
        train(data_zip, lang, data_set, max_seq_len, rewrite_files, lazy_loading)
    elif mode == 'test':
        assert model_dir, f'Need model_dir for test mode'
        predict_punct(model_dir, data_zip, lang, data_set, max_seq_len, outdir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='transformer baseline for SEPP-NLG 2021')
    parser.add_argument("data_zip", help="path to data zip file, e.g. 'data/sepp_nlg_2021_train_dev_data_v5.zip'")
    parser.add_argument("language", help="target language ('en', 'de', 'fr', 'it', or 'all'; i.e. one (or all) of the subfolders in the zip file's main folder)")
    parser.add_argument("eval_set", help="dataset to be evaluated (usually 'dev', 'test'), subfolder of 'lang'")
    parser.add_argument("outdir", help="folder to store predictions in, e.g. 'data/predictions' (language and dataset subfolders will be created automatically)")
    parser.add_argument("mode", help="train or test")
    parser.add_argument("model_dir", help="path to the trained model when testing", nargs="?", default="models/mbert_all/best_model/")
    parser.add_argument("rewrite_files", help="whether to re-write CONLL-style files", nargs="?", default=False)
    args = parser.parse_args()
    main(args.data_zip, args.language, args.eval_set, args.outdir, args.mode, model_dir=args.model_dir, rewrite_files=args.rewrite_files)