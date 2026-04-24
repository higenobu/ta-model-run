TARGET_DIR='output'

# Model parameters
BATCH_SIZE=32
MODEL_NAME_PATH='newtamodel'

# test data path (.txt file with news text in each line)
TEST_DATA_PATH='data'

python3 batch_predict.py \
	--data_path $TEST_DATA_PATH \
	--model_name_or_path $MODEL_NAME_PATH \
	--output_dir $TARGET_DIR \
	--batch_size $BATCH_SIZE \
