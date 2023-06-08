import numpy as np
import torch

from settings import SPLITS, MAX_LEN, MAX_CHAR_LEN

from util import token_level_to_char_level, get_test_loader2

from model import MyRobertaModel, MyCNN


def roberta_infer(data_loader, preds_start_, preds_end_):
	for k, data in enumerate(data_loader):
		input_ids = data['input_ids'].cuda(device=0)
		attention_mask = data['attention_mask'].cuda(device=0)
		text = data['text']
		offsets = data['offsets'].numpy()
		preds_start = np.zeros((input_ids.size(0), MAX_LEN))
		preds_end = np.zeros((input_ids.size(0), MAX_LEN))
		for i in range(1, SPLITS + 1):
			model = MyRobertaModel()
			model.cuda(device=0)
			model.load_state_dict(torch.load(f'../models/roberta_fold_{i}.pth'))
			model.eval()
			with torch.no_grad():
				start_logits, end_logits = model(input_ids, attention_mask)
				preds_start += torch.softmax(start_logits, dim=1).cpu().detach().numpy()
				preds_end += torch.softmax(end_logits, dim=1).cpu().detach().numpy()
		preds_start = preds_start / SPLITS
		preds_end = preds_end / SPLITS
		for i in range(len(input_ids)):
			s = token_level_to_char_level(text[i], offsets[i], preds_start[i])
			e = token_level_to_char_level(text[i], offsets[i], preds_end[i])
			preds_start_.append(s)
			preds_end_.append(e)
		if (k + 1) % 10 == 0:
			print('Batch [{}/{}] is done...'.format(k + 1, len(data_loader)))


def cnn_infer(df, x_test, preds_test_start, preds_test_end, len_voc):
	data_loader = get_test_loader2(
		df,
		np.array(x_test),
		np.array(preds_test_start),
		np.array(preds_test_end)
	)
	preds = []
	for k, data in enumerate(data_loader):
		preds_start = np.zeros((data['ids'].size(0), MAX_CHAR_LEN))
		preds_end = np.zeros((data['ids'].size(0), MAX_CHAR_LEN))
		for i in range(1, SPLITS + 1):
			model = MyCNN(len_voc)
			model.cuda(device=0)
			model.load_state_dict(torch.load(f'../models/cnn_fold_{i}.pth'))
			model.eval()
			with torch.no_grad():
				start_logits, end_logits = model(
					data['ids'].cuda(),
					data['sentiment_input'].cuda(),
					data['start_probs'].cuda(),
					data['end_probs'].cuda()
				)
				preds_start += torch.softmax(start_logits, dim=1).cpu().detach().numpy()
				preds_end += torch.softmax(end_logits, dim=1).cpu().detach().numpy()
		preds_start = preds_start / SPLITS
		preds_end = preds_end / SPLITS
		for i in range(len(data['ids'])):
			s = np.argmax(preds_start[i])
			e = np.argmax(preds_end[i])
			pred = data['text'][i][s:e] if s < e else data['text'][i]
			preds.append(pred)
		if (k + 1) % 10 == 0:
			print('Batch [{}/{}] is done...'.format(k + 1, len(data_loader)))
	return preds
