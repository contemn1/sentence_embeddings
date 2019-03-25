INPUT_DIR="/home/zxj/Documents"
INPUT_FILE_NAME="test_data.txt"
MODEL_ROOT="/media/zxj/sent_embedding_data"
WORD2VEC_PATH="$MODEL_ROOT/infersent/crawl-300d-2M.vec"
INFER_SENT_MODEL_PATH="$MODEL_ROOT/infersent/infersent{0}.pkl"
INFER_SENT_VERSION=2
GEN_SEN_MODEL_PATH="$MODEL_ROOT/gensen_models"
GEN_SEN_PREFIX="nli_large_bothskip"
GEN_SEN_EMBEDDING="$MODEL_ROOT/glove.840B.300d.h5"
SKIP_THOUGHT_PATH="$MODEL_ROOT/skip_thoughts_uni_2017_02_02"
SKIP_THOUGHT_MODEL_NAME="model.ckpt-501424"
QUICK_THOUGHT_CONFIG_PATH=$PWD/model_configs/MC-UMBC/eval.json
OUTPUT_DIR=$MODEL_ROOT/output

mkdir -p $OUTPUT_DIR

python encode_sentences.py --input-dir $INPUT_DIR --input-file-name $INPUT_FILE_NAME --word2vec-path $WORD2VEC_PATH \
--infer-sent-model-path $INFER_SENT_MODEL_PATH --infer-sent-version $INFER_SENT_VERSION \
--gensen-model-path $GEN_SEN_MODEL_PATH --gensen-prefix $GEN_SEN_PREFIX --gensen_embedding $GEN_SEN_EMBEDDING \
--skipthought-path $SKIP_THOUGHT_PATH --skipthought-model-name $SKIP_THOUGHT_MODEL_NAME \
--quick-thought-config-path $QUICK_THOUGHT_CONFIG_PATH --quick-thought-result-path $MODEL_ROOT \
--output-dir $OUTPUT_DIR