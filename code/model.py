import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel

from settings import PATH


class MyRobertaModel(nn.Module):
	def __init__(self):
		super(MyRobertaModel, self).__init__()
		
		config = RobertaConfig.from_pretrained(PATH + 'config.json', output_hidden_states=True)
		self.roberta_model = RobertaModel.from_pretrained(PATH + 'pytorch_model.bin', config=config)
		# random drop out
		self.dropout = nn.Dropout(0.5)
		self.fc = nn.Linear(self.roberta_model.config.hidden_size, 2)
		# initialize weight, bias
		nn.init.normal_(self.fc.weight, std=0.02)
		nn.init.normal_(self.fc.bias, 0)
	
	def forward(self, input_ids, attention_mask):
		# batch * MAX_LEN * hidden_size
		# x is hidden states of all layers
		# the two forms of output are last_hidden_state and pooler_output
		x = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
		# x.shape = 4 * batch * MAX_LEN * hidden_size
		# print(x[-1], x[-2], x[-3], x[-4])
		x = torch.stack([x[2][-1], x[2][-2], x[2][-3], x[2][-4]])
		# x.shape = batch * MAX_LEN * hidden_size
		x = torch.mean(x, 0)
		x = self.dropout(x)
		# x.shape = batch * MAX_LEN * 2
		x = self.fc(x)
		# x.shape = batch * MAX_LEN * 1
		x1, x2 = x.split(1, dim=-1)
		# x1.shape = batch * MAX_LEN
		x1, x2 = x1.squeeze(-1), x2.squeeze(-1)
		return x1, x2


class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same"):
		super(ConvBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
			nn.BatchNorm1d(out_channels),
			nn.ReLU(),
		)
	
	def forward(self, inputs):
		return self.conv(inputs)


class MyCNN(nn.Module):
	def __init__(self, len_voc, cnn_dim=32, char_embed_dim=16, sent_embed_dim=16, proba_cnn_dim=16, kernel_size=3):
		super(MyCNN, self).__init__()
		super().__init__()
		self.char_embeddings = nn.Embedding(len_voc, char_embed_dim)
		self.sentiment_embeddings = nn.Embedding(3, sent_embed_dim)
		self.probas_cnn = ConvBlock(2, proba_cnn_dim, kernel_size=kernel_size)
		self.cnn = nn.Sequential(
			ConvBlock(char_embed_dim + sent_embed_dim + proba_cnn_dim, cnn_dim, kernel_size=kernel_size),
			ConvBlock(cnn_dim, cnn_dim * 2, kernel_size=kernel_size),
			ConvBlock(cnn_dim * 2, cnn_dim * 4, kernel_size=kernel_size),
			ConvBlock(cnn_dim * 4, cnn_dim * 8, kernel_size=kernel_size),
		)
		self.logits = nn.Sequential(
			nn.Linear(cnn_dim * 8, cnn_dim),
			nn.ReLU(),
			nn.Linear(cnn_dim, 2),
		)
		self.high_dropout = nn.Dropout(p=0.5)
	
	def forward(self, tokens, sentiment, start_probas, end_probas):
		bs, t = tokens.size()
		probas = torch.cat([start_probas, end_probas], -1).permute(0, 2, 1)
		probas_fts = self.probas_cnn(probas).permute(0, 2, 1)
		char_fts = self.char_embeddings(tokens)
		sentiment_fts = self.sentiment_embeddings(sentiment).view(bs, 1, -1)
		sentiment_fts = sentiment_fts.repeat((1, t, 1))
		x = torch.cat([char_fts, sentiment_fts, probas_fts], -1).permute(0, 2, 1)
		features = self.cnn(x).permute(0, 2, 1)
		logits = self.logits(features)
		start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]
		return start_logits, end_logits
	
	
