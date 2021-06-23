"""
Evaluate submission zip file of SEPP-NLG 2021: https://sites.google.com/view/sentence-segmentation/
"""

__author__ = 'don.tuggener@zhaw.ch'

import io
import os
from typing import Optional, Union
from zipfile import ZipFile
from sklearn.metrics import classification_report


def main(data_zip: str, submission_zip: str, subtask: Optional[str] = 'both',  outdir: Optional[str] = '') -> None:

    with ZipFile(data_zip, 'r') as ground_truth:

        with ZipFile(submission_zip, 'r') as prediction:

            for lang in ['en', 'de', 'fr', 'it']:

                for data_set in ['dev', 'test', 'surprise_test']:

                    relevant_dir = os.path.join(lang, data_set)
                    ground_truth_tsv_files = [
                        fname for fname in ground_truth.namelist()
                        if relevant_dir in fname and fname.endswith('.tsv')
                    ]
                    prediction_tsv_files =[
                        fname for fname in prediction.namelist()
                        if relevant_dir in fname and fname.endswith('.tsv')
                    ]

                    all_task1_labels, all_task2_labels, all_task1_predictions, all_task2_predictions = list(), list(), list(), list()

                    for i, ground_truth_tsv_file in enumerate(ground_truth_tsv_files, 1):

                        print('\r' + str(i) + ' ' + ground_truth_tsv_file, end='')
                        basename = os.path.basename(ground_truth_tsv_file)
                        prediction_filename = os.path.join(lang, data_set, basename)
                        try:
                            prediction_file = next(fn for fn in prediction_tsv_files if fn.endswith(prediction_filename))
                        except StopIteration:
                            print('\nWARNING: ground file', prediction_filename, 'does not exist in predictions')
                            continue

                        with io.TextIOWrapper(ground_truth.open(ground_truth_tsv_file), encoding="utf-8") as f:
                            lines = f.read().strip().split('\n')
                            rows = [line.split('\t') for line in lines]
                            task1_labels, task2_labels = list(), list()
                            for row in rows:
                                task1_labels.append(row[1])
                                task2_labels.append(row[2])

                        with io.TextIOWrapper(prediction.open(prediction_file), encoding="utf-8") as f:
                            task1_predictions, task2_predictions = list(), list()
                            lines = f.read().strip().split('\n')
                            rows = [line.split('\t') for line in lines]
                            for row in rows:
                                if subtask in {'1', 'both'}:
                                    task1_predictions.append(row[1])
                                if subtask in {'2', 'both'}:
                                    task2_predictions.append(row[2])

                        if (subtask in {'1', 'both'} and not len(task1_labels) == len(task1_predictions)) \
                                or (subtask in {'2', 'both'} and not len(task2_labels) == len(task2_predictions)):
                            print(
                                f'\nWARNING: unequal no. of labels for files {ground_truth_tsv_file} and {prediction_filename}')
                            continue

                        all_task1_labels.extend(task1_labels)
                        all_task2_labels.extend(task2_labels)
                        all_task1_predictions.extend(task1_predictions)
                        all_task2_predictions.extend(task2_predictions)

                    print('\n' + submission_zip, lang, data_set)
                    outfile = os.path.basename(submission_zip).replace('.zip', '')
                    outfile = os.path.join(outdir, outfile)
                    if subtask in {'1', 'both'} and all_task1_predictions:
                        print('Subtask 1')
                        print(classification_report(all_task1_labels, all_task1_predictions, zero_division=0), file=open(f'{outfile}_{lang}_{data_set}_subtask1.txt', 'w'))
                    if subtask in {'2', 'both'} and all_task2_predictions:
                        print('Subtask 2')
                        print(classification_report(all_task2_labels, all_task2_predictions, zero_division=0), file=open(f'{outfile}_{lang}_{data_set}_subtask2.txt', 'w'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate submission zip file of SEPP-NLG 2021')
    parser.add_argument("data_zip", help="path to data zip file, e.g. 'data/sepp_nlg_2021_data.zip'")
    parser.add_argument("submission_zip", help="path to data zip file, e.g. 'TEAM_XY_sepp_nlg_2021.zip'")
    parser.add_argument("-s", "--subtask", help="which subtask to evaluate: 1, 2, both'", nargs="?", default="both")
    parser.add_argument("-o", "--outdir", help="folder for storing evaluation outputs'", nargs="?", default="")
    args = parser.parse_args()
    main(args.data_zip, args.submission_zip, subtask=args.subtask, outdir=args.outdir)
