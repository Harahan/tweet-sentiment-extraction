import torch

from settings import EPOCHS
from util import compute_jaccard_score, jaccard_from_logits_string


def train_model(model, data_dict, criterion, optimizer, filename):
	for epoch in range(EPOCHS):
		# set the model to train mode or eval mode
		for phase in ['train', 'valid']:
			if phase == 'train':
				model.train()
			else:
				model.eval()
			# initialize the loss and accuracy
			loss_train = 0
			acc_train = 0
			# train the model
			for k, data in enumerate(data_dict[phase]):
				# get the data
				input_ids = data['input_ids'].cuda(device=0)
				attention_mask = data['attention_mask'].cuda(device=0)
				offsets = data['offsets'].numpy()
				start_position = data['start_position'].cuda(device=0)
				end_position = data['end_position'].cuda(device=0)
				text = data['text']
				# clear the gradient
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):
					# forward
					start_logits, end_logits = model(input_ids, attention_mask)
					# compute the loss
					loss = criterion(start_logits, end_logits, start_position, end_position)
					if phase == 'train':
						# backward
						loss.backward()
						# update the parameters
						optimizer.step()
					# compute the accuracy
					start_position = start_position.cpu().detach().numpy()
					end_position = end_position.cpu().detach().numpy()
					start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
					end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
					input_ids = input_ids.cpu().detach().numpy()
					for i in range(len(input_ids)):
						acc = compute_jaccard_score(
							text[i],
							start_logits[i],
							end_logits[i],
							offsets[i],
							start_position[i],
							end_position[i]
						)
						acc_train += acc
					loss_train += loss.item() * len(input_ids)
			epoch_loss = loss_train / len(data_dict[phase].dataset)
			epoch_acc = acc_train / len(data_dict[phase].dataset)
			print('Epoch: [{}/{}], Phase: {}, Loss: {:.4f}, Acc: {:.4f}'.format(epoch + 1, EPOCHS, phase, epoch_loss, epoch_acc))
	# save the model
	torch.save(model.state_dict(), filename)


def train_model2(model, data_dict, criterion, optimizer, filename):
	for epoch in range(EPOCHS):
		# set the model to train mode or eval mode
		for phase in ['train', 'valid']:
			if phase == 'train':
				model.train()
			else:
				model.eval()
			# initialize the loss and accuracy
			loss_train = 0
			acc_train = 0
			# train the model
			for k, data in enumerate(data_dict[phase]):
				# clear the gradient
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):
					# forward
					start_logits, end_logits = model(
						data['ids'].cuda(),
						data['sentiment_input'].cuda(),
						data['start_probs'].cuda(),
						data['end_probs'].cuda()
					)
					# compute the loss
					loss = criterion(start_logits, end_logits, data['start'].cuda(device=0), data['end'].cuda(device=0))
					if phase == 'train':
						# backward
						loss.backward()
						# update the parameters
						optimizer.step()
					acc = jaccard_from_logits_string(
						data,
						start_logits,
						end_logits
					)
					acc_train += acc
					# update the loss and accuracy
					loss_train += loss.item() * len(data['ids'])
			epoch_loss = loss_train / len(data_dict[phase].dataset)
			epoch_acc = acc_train / len(data_dict[phase].dataset)
			print('Epoch: [{}/{}], Phase: {}, Loss: {:.4f}, Acc: {:.4f}'.format(epoch + 1, EPOCHS, phase, epoch_loss, epoch_acc))
	# save the model
	torch.save(model.state_dict(), filename)
	