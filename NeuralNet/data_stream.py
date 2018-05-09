import torch
from Settings.arguments import TexasHoldemAgrument


class DataStream:

	def __init__(self, data_path, batch_size):
		self.data_path = data_path

		self.train_input_data = None
		self.train_target_data = None
		self.train_mask_data = None
		self.train_data_count = None

		self.valid_input_data = None
		self.valid_target_data = None
		self.valid_mask_data = None
		self.valid_data_count = None

		self.batch_size = batch_size or TexasHoldemAgrument.batch_size

	def load_data(self, name):
		input_path = "%s%s.inputs" % (self.data_path, name)
		target_path = "%s%s.targets" % (self.data_path, name)
		mask_path = "%s%s.mask" % (self.data_path, name)
		data = [torch.load(path) for path in (input_path, target_path, mask_path)]
		return data

	def load_train_data(self, name="train"):
		data = self.load_data(name)
		self.train_input_data, self.train_target_data, self.train_mask_data = data
		self.train_data_count = self.train_input_data.size(0)
		assert self.train_input_data.size(0) - 1 == self.train_target_data.size(0) == self.train_mask_data.size(0)

	def load_valid_data(self, name="valid"):
		data = self.load_data(name)
		self.valid_input_data, self.valid_target_data, self.valid_mask_data = data
		self.valid_data_count = self.valid_input_data.size(0)
		assert self.train_input_data.size(0) - 1 == self.train_target_data.size(0) == self.train_mask_data.size(0)

	def get_train_batch_count(self):
		return self.train_input_data.size(0) // self.batch_size

	def get_valid_batch_count(self):
		return self.valid_input_data.size(0) // self.batch_size

	def start_epoch(self):
		shuffle = torch.randperm(self.train_data_count)
		self.train_input_data = self.train_input_data.index(shuffle)
		self.train_target_data = self.train_target_data.index(shuffle)
		self.train_mask_data = self.train_mask_data.index(shuffle)

		shuffle = torch.randperm(self.valid_data_count)
		self.valid_input_data = self.valid_input_data.index(shuffle)
		self.valid_target_data = self.valid_target_data.index(shuffle)
		self.valid_mask_data = self.valid_mask_data.index(shuffle)

	def get_train_batch(self, batch_index):
		start, end = batch_index * self.batch_size, (batch_index + 1) * self.batch_size
		return self.train_input_data[start:end, :],\
			self.train_target_data[start:end, :],\
			self.train_mask_data[start:end, :]

	def get_valid_batch(self, batch_index):
		start, end = batch_index * self.batch_size, (batch_index + 1) * self.batch_size
		return self.valid_input_data[start:end, :],\
			self.valid_target_data[start:end, :],\
			self.valid_mask_data[start:end, :]
