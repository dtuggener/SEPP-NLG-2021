#!/bin/bash

python run_adapter_transformers.py \
 --model_name_or_path bert-base-multilingual-uncased \
 --dataset_name hf_sepp_nlg_dataset_all \
 --output_dir $1 \
 --train_adapter \
 --do_train \
 --do_eval \
 --pad_to_max_length \
 # --overwrite_cache
# --model_name_or_path distilbert-base-uncased \



