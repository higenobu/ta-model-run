import torch
from torch.utils.data import DataLoader
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

def main(args):

	# Juman++の初期化
	jumanpp = Juman()

	# read the test data from the data_path as a list of sentences
	with open(f'{args.data_path}/test.txt', 'r') as f:
		sentences_test = f.readlines()
	sentences_test = [sentence.strip() for sentence in sentences_test]

	for text in sentences_test:
		with open(f'{args.output_dir}/preds.txt', 'a') as f:
			f.write("--------------------------------------------------\n")
			f.write(f"{text}\n")
			f.write("--------------------------------------------------\n")

			for i in range(1,8):
				model_name_or_path = f'{args.model_name_or_path}/{i}'
				config = AutoConfig.from_pretrained(model_name_or_path)
				tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
				model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
				words = ' '.join([mrph.midasi for mrph in jumanpp.analysis(text.replace('^', '＾')).mrph_list()])

				tokenized_inputs = tokenizer(words,
							padding=False, truncation=False, max_length=None,
							# We use this argument because the texts in our dataset are lists of words (with a label for each word).
							is_split_into_words=False,)

				model.eval()
				predictions = model(**{k: torch.tensor([v, v, v]) for k, v in tokenized_inputs.items()})[0]
				predictions_np = predictions.detach().numpy()

				# Remove extra dimensions and get the scalar value
				result = np.squeeze(predictions_np)[0]

				f.write(f"{titles[i-1]}: {result}\n")

			f.write("\n\n")

	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str, default='../data/press-assist_komura/data/news10000_final_data')
	parser.add_argument('--model_name_or_path', type=str, default='../notebooks/tmp/20220203-01/')
	parser.add_argument('--output_dir', type=str, default='../notebooks/tmp/20220203-01/')
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	main(args)