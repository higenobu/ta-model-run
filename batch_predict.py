from collections import OrderedDict
import torch
from pyknp import Juman
import numpy as np
from transformers import (
	AutoConfig,
	AutoModelForSequenceClassification,
	AutoTokenizer,
)
import argparse

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

titles = [
	'喜びを感じた',
	'恐怖を感じた',
	'驚きを感じた',
	'信頼できる情報と感じた',
	'曖昧な情報と感じた',
	'何かの意図をもって書かれたと感じた',
	'経済に期待がもてると感じた',
]

def batch_predict(sentences, model, tokenizer, device):
	# Tokenize all sentences in the batch
	tokenized_inputs = tokenizer(sentences, padding=True, return_tensors="pt")
	tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}

	# Make predictions
	with torch.no_grad():
		outputs = model(**tokenized_inputs)

	# model is regression-based
	predictions = outputs.logits.detach().cpu().numpy()

	# remove extra dimensions and get the scalar value
	predictions = np.squeeze(predictions)

	# return predictions as a list of floats
	return predictions.tolist()

def main(args):
	# Juman++ initialization
	jumanpp = Juman()

	# Read test data
	with open(f'{args.data_path}/test.txt', 'r') as f:
		sentences_test = [line.strip() for line in f.readlines()]

	# Write predictions
	with open(f'{args.output_dir}/preds_batch.txt', 'w') as f:
		net_preds = OrderedDict()

		for i in range(1,8):
			# Load model and tokenizer
			model_name_or_path = f'{args.model_name_or_path}/{i}'
			config = AutoConfig.from_pretrained(model_name_or_path)
			tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
			model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config).to(device)
			batch_size = args.batch_size  # Adjust batch size as needed

			model.eval()

			net_preds[i] = []

			# Process in batches
			for j in range(0, len(sentences_test), batch_size):
				batch_sentences = sentences_test[j:j + batch_size]
				batch_sentences = [' '.join([mrph.midasi for mrph in jumanpp.analysis(text.replace('^', '＾')).mrph_list()]) for text in batch_sentences]
				batch_predictions = batch_predict(batch_sentences, model, tokenizer, device)
				net_preds[i].extend(batch_predictions)

		for k, text in enumerate(sentences_test):
			f.write("--------------------------------------------------\n")
			f.write(f"{text}\n")
			f.write("--------------------------------------------------\n")

			for i in range(1,8):
				f.write(f"{titles[i-1]}: {net_preds[i][k]}\n")

			f.write("\n\n")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str, default='/home/matsu/psych_model_scripts/data')
	parser.add_argument('--model_name_or_path', type=str, default='/home/matsu/tamodels/')
	parser.add_argument('--output_dir', type=str, default='/home/matsu/psych_model_scripts/output')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	main(args)
