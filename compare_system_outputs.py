"""
Compare submission zip file of SEPP-NLG 2021: https://sites.google.com/view/sentence-segmentation/
according to method described in https://www.aclweb.org/anthology/E17-1018/, section 2, algorithm 1
"""

__author__ = 'don.tuggener@zhaw.ch'

import io
import os
import numpy as np
from typing import Optional, List, Union
from zipfile import ZipFile
from collections import Counter, defaultdict


def main(data_zip: str, submission_zip1: str, submission_zip2: str, languages: List[str] = ['en'],
         data_sets: List[str] = ['test'], outdir: Optional[str] = '') -> None:

    with ZipFile(data_zip, 'r') as ground_truth:

        with ZipFile(submission_zip1, 'r') as prediction1:

            with ZipFile(submission_zip2, 'r') as prediction2:

                for lang in languages:

                    for data_set in data_sets:

                        relevant_dir = os.path.join(lang, data_set)
                        ground_truth_tsv_files = [
                            fname for fname in ground_truth.namelist()
                            if relevant_dir in fname and fname.endswith('.tsv')
                        ]

                        prediction_tsv_files1 =[
                            fname for fname in prediction1.namelist()
                            if relevant_dir in fname and fname.endswith('.tsv')
                        ]

                        prediction_tsv_files2 =[
                            fname for fname in prediction2.namelist()
                            if relevant_dir in fname and fname.endswith('.tsv')
                        ]

                        all_task1_labels, all_task2_labels, all_task2_predictions1, all_task2_predictions2 = list(), list(), list(), list()
                        diff_counts = defaultdict(Counter)

                        for i, ground_truth_tsv_file in enumerate(ground_truth_tsv_files, 1):

                            print('\r' + str(i) + ' ' + ground_truth_tsv_file, end='')
                            basename = os.path.basename(ground_truth_tsv_file)
                            prediction_filename = os.path.join(lang, data_set, basename)
                            try:
                                prediction_file1 = next(fn for fn in prediction_tsv_files1 if fn.endswith(prediction_filename))
                                prediction_file2 = next(fn for fn in prediction_tsv_files2 if fn.endswith(prediction_filename))
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

                            with io.TextIOWrapper(prediction1.open(prediction_file1), encoding="utf-8") as f:
                                task2_predictions1 = list()
                                lines = f.read().strip().split('\n')
                                rows = [line.split('\t') for line in lines]
                                for row in rows:
                                   task2_predictions1.append(row[2])

                            with io.TextIOWrapper(prediction2.open(prediction_file2), encoding="utf-8") as f:
                                task2_predictions2 = list()
                                lines = f.read().strip().split('\n')
                                rows = [line.split('\t') for line in lines]
                                for row in rows:
                                   task2_predictions2.append(row[2])

                            if not len(task2_labels) == len(task2_predictions1) == len(task2_predictions2):
                                print(
                                    f'\nWARNING: unequal no. of labels for files {ground_truth_tsv_file} and {prediction_file1}, and {prediction_file2}')
                                continue

                            for gt, p1, p2 in zip(task2_labels, task2_predictions1, task2_predictions2):
                                if not p1 == p2:
                                    if gt == p1:
                                        diff_class = 'new error'
                                    elif gt == p2:
                                        diff_class = 'correction'
                                    else:
                                        diff_class = 'changed error'
                                    diff_counts[gt][diff_class] += 1

                        print()
                        sum_diffs = sum(np.ravel([list(d.values()) for d in diff_counts.values()]))
                        for label, diffs in diff_counts.items():
                            # % of total diff - normalize regarding absolute counts?
                            print(f'{label} ({sum(diffs.values())} {round(100 * sum(diffs.values()) / sum_diffs, 2)}%)')
                            sum_diff = sum(diffs.values())
                            for diff, cnt in diffs.most_common():
                                print(f'{diff}: {cnt} ({round(100 * cnt/sum_diff, 2)}%)')
                            print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate submission zip file of SEPP-NLG 2021')
    parser.add_argument("data_zip", help="path to data zip file, e.g. 'data/sepp_nlg_2021_data.zip'")
    parser.add_argument("submission_zip1", help="path to data zip file, e.g. 'TEAM_XY_sepp_nlg_2021.zip'")
    parser.add_argument("submission_zip2", help="path to data zip file, e.g. 'TEAM_XY_sepp_nlg_2021.zip'")
    parser.add_argument("-o", "--outdir", help="folder for storing evaluation outputs'", nargs="?", default="")
    args = parser.parse_args()
    main(args.data_zip, args.submission_zip1, args.submission_zip2, outdir=args.outdir)
