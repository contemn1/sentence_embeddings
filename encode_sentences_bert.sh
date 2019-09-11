#!/usr/bin/env bash
INPUT_DIR="/home/zxj/Data/sentence_analogy_datasets"
MODEL_ROOT="/home/zxj/Data/sent_embedding_data"
OUTPUT_DIR=$MODEL_ROOT/output

mkdir -p $OUTPUT_DIR

for INPUT_PATH in $INPUT_DIR/*.txt
do
  python bert_sentence_encoder.py --input-path $INPUT_PATH \
  --batch-size 64 --use-cuda --with-special-tokens --output-dir $OUTPUT_DIR
done