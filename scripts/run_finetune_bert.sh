USER=${1:-"phuongnm"}
MODEL_NAME=${2:-"../settings/phobert-t3042/models/"}  # vinai/phobert-base
ROOT_DIR=${3:-"../"}  
DATA_DIR=${4:-"data/zalo-tfidf30/"}  
SETTING_NAME=${5:-"phobert-t30"}  
NUM_EPOCH=${6:-5}  
LR=${7:-1e-5}  
NUM_LABEL=${8:-2}  
MAX_LEN=${9:-512}  

CODE_DIR="${ROOT_DIR}/src/" 
DATA_DIR="${ROOT_DIR}/$DATA_DIR"

SCRIPT_DIR=$(pwd)

for iSEED in {42..42}
do
  SETTING_NAME_SEED=${SETTING_NAME}${iSEED}
  SETTING_DIR="${ROOT_DIR}/settings/${SETTING_NAME_SEED}/" 
  MODEL_OUT="${SETTING_DIR}/models"
  mkdir $SETTING_DIR
  cp ${SCRIPT_DIR}/run_train.sh   $SETTING_DIR

  
  cd $CODE_DIR  && python3 ./run_glue.py \
  --model_name_or_path $MODEL_NAME \
  --do_train \
  --eval_steps 200 \
  --do_predict \
  --num_label  ${NUM_LABEL} \
  --seed ${iSEED} \
  --train_file $DATA_DIR/train.csv \
  --validation_file $DATA_DIR/dev.csv \
  --test_file $DATA_DIR/test.csv \
  --max_seq_length $MAX_LEN \
  --per_device_train_batch_size 16 \
  --learning_rate $LR \
  --warmup_steps 0 \
  --num_train_epochs $NUM_EPOCH \
  --save_total_limit 1 \
  --logging_dir $MODEL_OUT/tensorboard --logging_steps 200 \
  --output_dir $MODEL_OUT --overwrite_output_dir \
  |tee $SETTING_DIR/train.log
done
