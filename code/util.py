# pre save the tokens of positive, negative and neutral
import os
import random
import numpy as np
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from torch import nn

from settings import SEED, PATH, BATCH_SIZE, DEVICE_SIZE, MAX_LEN


def seed_everything():
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	os.environ['PYTHONHASHSEED'] = str(SEED)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(SEED)
		torch.cuda.manual_seed_all(SEED)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = True


sentiment_ids = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
# build tokenizer
# use BPE encoder to encode the text
# add prefix space to the text, so that the tokenizer can recognize the first word
tokenizer = ByteLevelBPETokenizer(
	vocab=PATH + 'vocab.json',
	merges=PATH + 'merges.txt',
	lowercase=True,
	add_prefix_space=True
)


def get_train_val_loaders(df, train_idx, val_idx):
	train_df = df.iloc[train_idx]
	val_df = df.iloc[val_idx]
	
	from data import TweetDataset
	train_loader = torch.utils.data.DataLoader(
		TweetDataset(train_df),
		batch_size=BATCH_SIZE * DEVICE_SIZE,
		shuffle=True,
		num_workers=2,
		drop_last=True
	)
	val_loader = torch.utils.data.DataLoader(
		TweetDataset(val_df),
		batch_size=BATCH_SIZE * DEVICE_SIZE,
		shuffle=False,
		num_workers=2,
		drop_last=False
	)
	
	data_dict = {'train': train_loader, 'valid': val_loader}
	return data_dict


def get_test_loader(df):
	from data import TweetDataset
	test_loader = torch.utils.data.DataLoader(
		TweetDataset(df),
		batch_size=BATCH_SIZE * DEVICE_SIZE,
		shuffle=False,
		num_workers=2,
		drop_last=False
	)
	return test_loader


def get_selected_text(text, start_position, end_position, offsets):
	selected_text = ""
	for i in range(start_position, end_position + 1):
		# ? why add space
		selected_text += text[offsets[i][0]: offsets[i][1]]
		if (i + 1) < len(offsets) and offsets[i][1] < offsets[i + 1][0]:
			selected_text += " "
	return selected_text


def loss_fn(start_logits, end_logits, start_position, end_position):
	ce_loss = nn.CrossEntropyLoss(label_smoothing=1 / MAX_LEN)
	# input: (N, C) where C = number of classes -> (batch, MAX_LEN)
	# target: (N) where each value is 0 <= targets[i] <= C-1 -> (batch)
	# output: scalar -> (1)
	start_loss = ce_loss(start_logits, start_position)
	end_loss = ce_loss(end_logits, end_position)
	total_loss = start_loss + end_loss
	return total_loss


def jaccard(str1, str2):
	a = set(str1.lower().split())
	b = set(str2.lower().split())
	if (len(a) == 0) & (len(b) == 0):
		return 0.5
	c = a.intersection(b)
	return float(len(c)) / (len(a) + len(b) - len(c))


def compute_jaccard_score(text, start_logits, end_logits, offsets, start_position, end_position):
	start_pred = np.argmax(start_logits)
	end_pred = np.argmax(end_logits)
	pred = text if start_pred > end_pred else get_selected_text(text, start_pred, end_pred, offsets)
	selected_text = get_selected_text(text, start_position, end_position, offsets)
	return jaccard(selected_text, pred)


def token_level_to_char_level(text, offsets, preds):
	probas_char = np.zeros(len(text))
	for i, offset in enumerate(offsets):
		if offset[0] or offset[1]:
			probas_char[offset[0]:offset[1]] = preds[i]
	return probas_char


def get_train_val_loaders2(df, x, probs_start, probs_end, train_idx, val_idx):
	train_df = df.iloc[train_idx]
	val_df = df.iloc[val_idx]
	train_x = x[train_idx]
	val_x = x[val_idx]
	train_probs_start = probs_start[train_idx]
	train_probs_end = probs_end[train_idx]
	val_probs_start = probs_start[val_idx]
	val_probs_end = probs_end[val_idx]
	from data import TweetDataset2
	train_loader = torch.utils.data.DataLoader(
		TweetDataset2(train_df, train_x, train_probs_start, train_probs_end),
		batch_size=BATCH_SIZE * DEVICE_SIZE,
		shuffle=True,
		num_workers=2,
		drop_last=True
	)
	val_loader = torch.utils.data.DataLoader(
		TweetDataset2(val_df, val_x, val_probs_start, val_probs_end),
		batch_size=BATCH_SIZE * DEVICE_SIZE,
		shuffle=False,
		num_workers=2,
		drop_last=False
	)
	data_dict = {'train': train_loader, 'valid': val_loader}
	return data_dict


def get_test_loader2(df, x, probs_start, probs_end):
	from data import TweetDataset2
	test_loader = torch.utils.data.DataLoader(
		TweetDataset2(df, x, probs_start, probs_end),
		batch_size=BATCH_SIZE * DEVICE_SIZE,
		shuffle=False,
		num_workers=2,
		drop_last=False
	)
	return test_loader


def jaccard_from_logits_string(data, start_logits, end_logits):
	n = start_logits.size(0)
	score = 0
	start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
	end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
	for i in range(n):
		start_idx = np.argmax(start_logits[i])
		end_idx = np.argmax(end_logits[i])
		text = data["text"][i]
		pred = text[start_idx: end_idx] if start_idx < end_idx else text[end_idx: start_idx]
		score += jaccard(data["selected_text"][i], pred)
	return score
