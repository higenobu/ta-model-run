TARGET_DIR='newtamodel'

# Model parameters
MAX_LEN=256
BATCH_SIZE=32
LEARNING_RATE=2.9051435624508314e-06
EPOCHS=4
MODEL_NAME_PATH='nlp-waseda/roberta-base-japanese'

# new data path
DATA_PATH='data/old'

# Data/Model download?

# aws s3 cp s3://psych-dl-model/press-assist/{TARGET_DIR}/1/  tmp/{TARGET_DIR}/1/ --recursive --exclude "checkpoint*/**"
# aws s3 cp s3://psych-dl-model/press-assist/{TARGET_DIR}/2/  tmp/{TARGET_DIR}/2/ --recursive --exclude "checkpoint*/**"
# aws s3 cp s3://psych-dl-model/press-assist/{TARGET_DIR}/3/  tmp/{TARGET_DIR}/3/ --recursive --exclude "checkpoint*/**"
# aws s3 cp s3://psych-dl-model/press-assist/{TARGET_DIR}/4/  tmp/{TARGET_DIR}/4/ --recursive --exclude "checkpoint*/**"
# aws s3 cp s3://psych-dl-model/press-assist/{TARGET_DIR}/5/  tmp/{TARGET_DIR}/5/ --recursive --exclude "checkpoint*/**"
# aws s3 cp s3://psych-dl-model/press-assist/{TARGET_DIR}/6/  tmp/{TARGET_DIR}/6/ --recursive --exclude "checkpoint*/**"
# aws s3 cp s3://psych-dl-model/press-assist/{TARGET_DIR}/7/  tmp/{TARGET_DIR}/7/ --recursive --exclude "checkpoint*/**"


python3 tatrain.py \
	--data_path $DATA_PATH \
	--max_len $MAX_LEN \
	--batch_size $BATCH_SIZE \
	--learning_rate $LEARNING_RATE \
	--epochs $EPOCHS \
	--output_dir $TARGET_DIR \
	--model_name_or_path $MODEL_NAME_PATH \


# python predict.py \
# 	--data_path {DATA_PATH} \
# 	--model_name_or_path {MODEL_NAME_PATH} \
# 	--output_dir tmp/{TARGET_DIR}/output \
