import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import (
	AutoConfig,
	AutoModelForSequenceClassification,
	AutoTokenizer,
)
from sklearn.metrics import mean_squared_error
from scipy import stats
#import optuna

from tqdm import tqdm
from copy import deepcopy
import argparse

from utils_bk import EconIndicatorDataset
from utils_bk import process_data_pipeline

# GPU check
if torch.cuda.is_available():
	device = torch.device("cuda")
	print("GPU is available")
else:
	device = torch.device("cpu")
	print("GPU is not available")
print(device)

titles = [
    '喜びを感じた',
    '恐怖を感じた',
    '驚きを感じた',
    '信頼できる情報と感じた',
    '曖昧な情報と感じた',
    '何かの意図をもって書かれたと感じた',
    '経済に期待がもてると感じた',
'8-score',
'9-score',
'10-score',
]

def main(args):

	sentences_train_old, labels_train_old, sentences_dev_old, labels_dev_old, sentences_test_old, labels_test_old = process_data_pipeline(args.data_path)

	df_list_trains = []
	df_list_devs = []
	df_list_tests = []

	for i in range(1, 8):
		df_list_trains.append(pd.DataFrame({"text": sentences_train_old[i], "label": labels_train_old[i]}))
		df_list_devs.append(pd.DataFrame({"text": sentences_dev_old[i], "label": labels_dev_old[i]}))
		df_list_tests.append(pd.DataFrame({"text": sentences_test_old[i], "label": labels_test_old[i]}))

	# add an empty element to the beginning of the list
	df_list_trains.insert(0, None)
	df_list_devs.insert(0, None)
	df_list_tests.insert(0, None)

	for i in range(1,8):
		model_name_or_path = f'{args.model_name_or_path}'
		config = AutoConfig.from_pretrained(model_name_or_path)
		tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
		model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)

		# Dat(aFrame dfを訓練データと評価データに分割
		train_df = df_list_trains[i]
		val_df = df_list_devs[i]

		# Datasetインスタンスを作成
		train_dataset = EconIndicatorDataset(train_df, tokenizer, args.max_len)
		val_dataset = EconIndicatorDataset(val_df, tokenizer, args.max_len)

		# DataLoaderにロード
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = model.to(device)

		optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
		best_model = None
		best_val_loss = float('inf')

		for epoch in range(args.epochs):
			model.train()
			for _,data in tqdm(enumerate(train_loader), total=len(train_loader)):
				ids = data['input_ids'].to(device, dtype=torch.long)
				mask = data['attention_mask'].to(device, dtype=torch.long)
				targets = data['labels'].to(device, dtype=torch.float)

				outputs = model(ids, mask)
				loss = torch.nn.MSELoss()(outputs.logits.view(-1), targets)


				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			model.eval()
			with torch.no_grad():
				total_loss = 0
				total_size = 0
				for _, data in tqdm(enumerate(val_loader), total=len(val_loader)):
					ids = data['input_ids'].to(device, dtype=torch.long)
					mask = data['attention_mask'].to(device, dtype=torch.long)
					targets = data['labels'].to(device, dtype=torch.float)

					outputs = model(ids, mask)
					loss = torch.nn.MSELoss()(outputs.logits.view(-1), targets)

					total_loss += loss.item() * targets.size(0)
					total_size += targets.size(0)

				avg_val_loss = total_loss / total_size
				print(f"Validation Loss after epoch {epoch+1}: {avg_val_loss}")
				# Check if this is the best model so far
				if avg_val_loss < best_val_loss:
					best_val_loss = avg_val_loss
					best_model = deepcopy(model)

		# DataFrame dfを訓練データと評価データに分割
		test_df = df_list_tests[i]

		# Datasetインスタンスを作成
		test_dataset = EconIndicatorDataset(test_df, tokenizer, args.max_len)

		# DataLoaderにロード
		test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

		test_predictions = []
		with torch.no_grad():
			for _, data in tqdm(enumerate(test_loader), total=len(test_loader)):
				ids = data['input_ids'].to(device, dtype=torch.long)
				mask = data['attention_mask'].to(device, dtype=torch.long)

				outputs = best_model(ids, mask)
				test_predictions.extend(outputs.logits.view(-1).detach().cpu().numpy())
		# Add the predictions as a new column to the test DataFrame
		df_list_tests[i]['predict'] = test_predictions

		model.eval()
		# モデルの重みを保存
		model_path = f'{args.output_dir}/{i}/pytorch_model.bin'

		print(model_path)
		# overwrite
		if os.path.exists(model_path):
			os.system(f'rm -r {args.output_dir}/{i}')
		os.makedirs(f'{args.output_dir}/{i}')
		os.makedirs(f'{args.output_dir}/{i}/hf_model')

		# torch.save(best_model.to('cpu').state_dict(), model_path)
		best_model.save_pretrained(f'{args.output_dir}/{i}/')
		tokenizer.save_pretrained(f'{args.output_dir}/{i}/')

	print("--------------------------------------------------")
	print("Start evaluation", end="\n\n")

	for i in range(1,10):

		print("--------------------------------------------------")
		print(f"Evaluating for Title: {titles[i-1]}")

		actual_values = df_list_tests[i]["label"]
		predict_values = df_list_tests[i]["predict"]

		#RMSEの計算
		rmse = np.sqrt(mean_squared_error(actual_values, predict_values))
		print(i,f"rmse:{rmse}")
		# MAEの計算
		mae = np.mean(np.abs(predict_values - actual_values))
		print(i,f"mae:{mae}")
		# ピアソンの相関係数の計算
		correlation = np.corrcoef(actual_values, predict_values)[0, 1]
		print(i, f"correlation(pearson): {correlation}")
		#スピアマンの順位相関係数
		correlation, _ = stats.spearmanr(actual_values, predict_values)
		print(i, f"correlation(spearman): {correlation}")

		print("--------------------------------------------------", end="\n\n")

	#--------------------------------------------------

	def objective(trial):
		# ハイパーパラメータの探索範囲を指定
		learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
		batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
		epochs = trial.suggest_int('epochs', 1, 10)

		# モデルの再初期化と設定
		model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
		model = model.to(device)

		# オプティマイザーの設定
		optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

		# 訓練ループ
		for epoch in range(epochs):
			model.train()
			for _,data in tqdm(enumerate(train_loader), total=len(train_loader)):
				ids = data['input_ids'].to(device, dtype=torch.long)
				mask = data['attention_mask'].to(device, dtype=torch.long)
				targets = data['labels'].to(device, dtype=torch.float)

				outputs = model(ids, mask)
				loss = torch.nn.MSELoss()(outputs.logits.view(-1), targets)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		# 評価ループ
		model.eval()
		with torch.no_grad():
			total_loss = 0
			total_size = 0
			for _, data in tqdm(enumerate(val_loader), total=len(val_loader)):
				ids = data['input_ids'].to(device, dtype=torch.long)
				mask = data['attention_mask'].to(device, dtype=torch.long)
				targets = data['labels'].to(device, dtype=torch.float)

				outputs = model(ids, mask)
				loss = torch.nn.MSELoss()(outputs.logits.view(-1), targets)

				total_loss += loss.item() * targets.size(0)
				total_size += targets.size(0)

			avg_val_loss = total_loss / total_size
		# 最適化する値を返す（例: バリデーション損失）
		return avg_val_loss

	'''
	for i in range(1,2):
		model_name_or_path = f'tmp/{TARGET_DIR}/{i}'
		config = AutoConfig.from_pretrained(model_name_or_path)
		tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
		#model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)


		study = optuna.create_study(direction='minimize')  # 最小化問題として設定
		study.optimize(objective, n_trials=100)  # 最大100回の試行

		# 最適なハイパーパラメータと目的関数の値を表示
		print('Best trial:', study.best_trial.params)
		print('Best value:', study.best_value)
	'''

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	TOKENIZERS_PARALLELISM=True
	# Define arguments
	parser.add_argument("--model_name_or_path", default='latest', type=str, help="Path to the model")
	parser.add_argument("--output_dir", default='tmp', type=str, help="Path to the output directory")
	parser.add_argument("--data_path", default='data', type=str, help="Path to the data directory")
	parser.add_argument("--max_len", default=256, type=int, help="Maximum length")
	parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
	parser.add_argument("--learning_rate", default=2.9051435624508314e-06, type=float, help="Learning rate")
	parser.add_argument("--epochs", default=4, type=int, help="Number of epochs")

	# Parse arguments
	args = parser.parse_args()

	main(args)
