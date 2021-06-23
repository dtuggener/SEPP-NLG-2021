#!/bin/bash

python run_adapter_transformers.py \
 --model_name_or_path bert-base-multilingual-uncased \
 --tokenizer bert-base-multilingual-uncased \
 --dataset_name hf_sepp_nlg_dataset_all \
 --load_adapter $1 \
 --output_dir $2 \
 --do_predict \
 #--overwrite_cache
# --model_name_or_path distilbert-base-uncased \
# --tokenizer distilbert-base-uncased \


