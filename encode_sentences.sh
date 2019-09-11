#!/usr/bin/env bash
INPUT_DIR="/home/zxj/Data/sentence_analogy_datasets"
MODEL_ROOT="/home/zxj/Data/sent_embedding_data"
GEN_SEN_MODEL_PATH="$MODEL_ROOT/gensen_models"
GEN_SEN_PREFIX="nli_large_bothskip"
GEN_SEN_EMBEDDING="$MODEL_ROOT/glove.840B.300d.h5"
SKIP_THOUGHT_PATH="$MODEL_ROOT/skip_thoughts_uni_2017_02_02"
SKIP_THOUGHT_MODEL_NAME="model.ckpt-501424"
QUICK_THOUGHT_CONFIG_PATH="/home/zxj/PycharmProjects/sentence_embeddings/model_configs/MC-UMBC/eval.json"
OUTPUT_DIR=$MODEL_ROOT/output

mkdir -p $OUTPUT_DIR

for INPUT_PATH in $INPUT_DIR/*.txt
do
  python encode_sentences.py --input-path $INPUT_PATH \
--model-dir $MODEL_ROOT --infer-sent-version 1 2 \
--gensen-model-path $GEN_SEN_MODEL_PATH --gensen-prefix $GEN_SEN_PREFIX --gensen_embedding $GEN_SEN_EMBEDDING \
--skipthought-path $SKIP_THOUGHT_PATH --skipthought-model-name $SKIP_THOUGHT_MODEL_NAME \
--quick-thought-config-path $QUICK_THOUGHT_CONFIG_PATH --quick-thought-result-path $MODEL_ROOT \
--output-dir $OUTPUT_DIR
done