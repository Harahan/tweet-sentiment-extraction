import numpy as np
import torch

from settings import MAX_LEN, MAX_CHAR_LEN
from util import tokenizer, sentiment_ids
from keras.utils import pad_sequences


class TweetDataset(torch.utils.data.Dataset):
	def __init__(self, df):
		self.df = df
		self.selected = 'selected_text' in df
	
	def __getitem__(self, i):
		
		def extract_data(r):
			text = " " + " ".join(r.text.lower().split())
			encoding = tokenizer.encode(text)
			sentiment_id = sentiment_ids[r.sentiment]
			input_ids = [0] + [sentiment_id] + [2, 2] + encoding.ids + [2]
			offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]
			
			pad_len = MAX_LEN - len(input_ids)
			# 1: padding token, 0: start token, 2: end token
			if pad_len > 0:
				input_ids += [1] * pad_len
				offsets += [(0, 0)] * pad_len
			
			input_ids = torch.tensor(input_ids)
			#  1 indicate a value that should be attended to while 0 indicate a padded value.
			attention_mask = torch.where(input_ids != 1, torch.tensor(1), torch.tensor(0))
			offsets = torch.tensor(offsets)
			return input_ids, attention_mask, text, offsets
		
		def get_tar_off(row, text, offests):
			selected_text = " " + " ".join(row.selected_text.lower().split())
			l = len(selected_text) - 1
			start_position, end_position = -1, -1
			for i in range(len(text)):
				if " " + text[i: i + l] == selected_text:
					start_position, end_position = i, i + l - 1
					break
			chars = [0] * len(text)
			for i in range(start_position, end_position + 1):
				chars[i] = 1
			targets = []
			for i, (x, y) in enumerate(offsets):
				if sum(chars[x: y]) > 0:
					targets.append(i)
			return targets[0], targets[-1]
		
		r = self.df.iloc[i]
		# offset: the start and end position of each token in the original text
		input_ids, attention_mask, text, offsets = extract_data(r)
		
		d = {
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'text': text,
			'offsets': offsets,
		}
		
		# position is to tokens
		if self.selected:
			start_position, end_position = get_tar_off(r, text, offsets)
			d['start_position'] = start_position
			d['end_position'] = end_position
		return d
	
	def __len__(self):
		return len(self.df)


class TweetDataset2(torch.utils.data.Dataset):
	def __init__(self, df, x, start_probs, end_probs):
		
		def get_start_end(text, selected_text):
			len_s = len(selected_text)
			len_t = len(text)
			s, e = 0, 0
			for i in range(len_t):
				if len_s + i <= len_t and text[i:i + len_s] == selected_text:
					s, e = i, i + len_s
					break
			return s, e
		
		self.df = df
		self.x = pad_sequences(x, maxlen=MAX_CHAR_LEN, padding='post', truncating='post')
		self.selected = 'selected_text' in df
		self.start_probs = np.zeros((len(df), MAX_CHAR_LEN), dtype=float)
		for i, p in enumerate(start_probs):
			l = min(len(p), MAX_CHAR_LEN)
			self.start_probs[i,:l] = p[:l]
		self.start_probs = np.expand_dims(self.start_probs, axis=2)
		self.end_probs = np.zeros((len(df), MAX_CHAR_LEN), dtype=float)
		for i, p in enumerate(end_probs):
			l = min(len(p), MAX_CHAR_LEN)
			self.end_probs[i,:l] = p[:l]
		self.end_probs = np.expand_dims(self.end_probs, axis=2)
		
		self.texts = df['text'].values
		
		if self.selected:
			self.selected_texts = df['selected_text'].values
		else:
			self.selected_texts = [''] * len(df)
		
		sentiments_id = {'positive': 0, 'neutral': 1, 'negative': 2}
		self.sentiments = df['sentiment'].values
		self.sentiments_input = [sentiments_id[s] for s in self.sentiments]
		
		if self.selected:
			self.start_idx = []
			self.end_idx = []
			for text, selected_text in zip(df['text'].values, df['selected_text'].values):
				s, e = get_start_end(text, selected_text.strip())
				self.start_idx.append(s)
				self.end_idx.append(e)
		else:
			self.start_idx = [0] * len(df)
			self.end_idx = [0] * len(df)
	
	def __getitem__(self, i):
		return {
			'ids': torch.tensor(self.x[i], dtype=torch.long),
			'start_probs': torch.tensor(self.start_probs[i]).float(),
			'end_probs': torch.tensor(self.end_probs[i]).float(),
			'start': torch.tensor(self.start_idx[i], dtype=torch.long),
			'end': torch.tensor(self.end_idx[i], dtype=torch.long),
			'text': self.texts[i],
			'selected_text': self.selected_texts[i],
			'sentiment': self.sentiments[i],
			'sentiment_input': torch.tensor(self.sentiments_input[i])
		}
	
	def __len__(self):
		return len(self.df)

