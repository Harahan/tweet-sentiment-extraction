import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from infer import roberta_infer, cnn_infer
from train import train_model, train_model2
from model import MyRobertaModel, MyCNN
from settings import SPLITS, LR, SEED
from util import seed_everything, loss_fn, get_train_val_loaders, get_test_loader, get_train_val_loaders2

if __name__ == "__main__":
	warnings.filterwarnings('ignore')
	seed_everything()
	train = pd.read_csv('../data/train.csv')
	# drop na
	train.dropna(inplace=True)
	train.reset_index(drop=True, inplace=True)
	skf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=SEED)
	
	# ----------------- First Stage -----------------
	for fold, (train_idx, valid_idx) in enumerate(skf.split(train, train.sentiment), start=1):
		print('#' * 20, 'Start Fold Training: ', fold, ' ', '#' * 20)
		torch.cuda.empty_cache()
		model = MyRobertaModel()
		# model = torch.nn.DataParallel(model, device_ids=[i for i in range(DEVICE_SIZE)])
		model = model.cuda(device=0)
		optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
		criterion = loss_fn
		# get the data
		data_dict = get_train_val_loaders(train, train_idx, valid_idx)
		train_model(
			model,
			data_dict,
			criterion,
			optimizer,
			'../models/roberta_fold_' + str(fold) + '.pth'
		)
		print('#' * 20, 'Finish Fold Training: ', fold, ' ', '#' * 20, '\n')
	# data preparation for second stage
	data_loader = get_test_loader(train)
	preds_train_start = []
	preds_train_end = []
	roberta_infer(data_loader, preds_train_start, preds_train_end)
	test = pd.read_csv('../data/test.csv')
	test_loader = get_test_loader(test)
	preds_test_start = []
	preds_test_end = []
	roberta_infer(test_loader, preds_test_start, preds_test_end)
	tokenizer_char = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
	tokenizer_char.fit_on_texts(train['text'].values)
	len_voc = len(tokenizer_char.word_index) + 1
	x_train = tokenizer_char.texts_to_sequences(train['text'].values)
	x_test = tokenizer_char.texts_to_sequences(test['text'].values)
	skf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=SEED)
	
	# ----------------- Second Stage -----------------
	for fold, (train_idx, valid_idx) in enumerate(skf.split(train, train.sentiment), start=1):
		print('#' * 20, 'Start Fold Training: ', fold, ' ', '#' * 20)
		torch.cuda.empty_cache()
		model = MyCNN(len_voc)
		# model = torch.nn.DataParallel(model, device_ids=[i for i in range(DEVICE_SIZE)])
		model = model.cuda(device=0)
		optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
		criterion = loss_fn
		# get the data
		data_dict = get_train_val_loaders2(
			train,
			np.array(x_train),
			np.array(preds_train_start),
			np.array(preds_train_end),
			train_idx,
			valid_idx
		)
		train_model2(
			model,
			data_dict,
			criterion,
			optimizer,
			'../models/cnn_fold_' + str(fold) + '.pth'
		)
		print('#' * 20, 'Finish Fold Training: ', fold, ' ', '#' * 20, '\n')
	# data inference for second stage
	preds = cnn_infer(test, x_test, preds_test_start, preds_test_end, len_voc)
	df = pd.read_csv('../data/sample_submission.csv')
	df['selected_text'] = preds
	df['selected_text'] = df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split()) == 1 else x)
	df['selected_text'] = df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split()) == 1 else x)
	df['selected_text'] = df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split()) == 1 else x)
	df.to_csv('../data/submission_stage2.csv', index=False)
	
	