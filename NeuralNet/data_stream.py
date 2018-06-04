import os
import torch
import pickle
from Settings.arguments import TexasHoldemArgument as Argument


class DataStream:

    def __init__(self):
        """initialize"""
        self.src_folder = "../Data/TrainingSamples/Texas/BucketValue/"
        self.data_count_per_file = 100
        self.bucket_count = 500
        self.total_data_count = None

        self.total_inputs = None
        self.total_targets = None
        self.total_masks = None

        self.train_count = None
        self.train_input_data = None
        self.train_target_data = None
        self.train_mask_data = None
        self.train_batch_size = Argument.batch_size

        self.valid_count = None
        self.valid_input_data = None
        self.valid_target_data = None
        self.valid_mask_data = None
        self.valid_batch_size = Argument.batch_size

        self.file_names = os.listdir(self.src_folder)
        self.file_count = len(self.file_names)
        self.file_paths = [self.src_folder + file_name for file_name in self.file_names]

    def load_data(self):
        """load inputs(X), targets(Y) and mask
        load every file in self.src_folder merge them into a single FloatTensor
        save all inputs in self.total_inputs
        save all targets ins self.total_targets
        save all masks in self.total_masks
        """
        total_data_count = self.file_count * self.data_count_per_file
        self.total_data_count = total_data_count
        self.total_inputs = Argument.Tensor(total_data_count, self.bucket_count * 2 + 1)
        self.total_targets = Argument.Tensor(total_data_count, self.bucket_count * 2)
        self.total_masks = Argument.Tensor(total_data_count, self.bucket_count * 2)

        current_idx = 0

        for file_path in self.file_paths:
            with open(file_path, "rb") as f:
                inputs, targets, mask = pickle.load(f)
                self.total_inputs[current_idx:current_idx + self.data_count_per_file, :] = inputs
                self.total_targets[current_idx:current_idx + self.data_count_per_file, :] = targets
                self.total_masks[current_idx:current_idx + self.data_count_per_file, :] = mask.repeat(1, 2)
                current_idx += self.data_count_per_file

    def split_data(self, fraction=0.9):
        """split total_inputs, total_targets, total_masks into training set and validation set
        split data would be save in self.train_xxx_data and self.valid_xxx_data respectively
        before splitting, data should be randomly permuted
        should be called after load_data()
        :parameter fraction: fraction for training, set it to 0.9 by default
        :return: None
        """
        shuffle = torch.randperm(self.total_data_count)
        self.total_inputs = self.total_inputs.index((shuffle, ))
        self.total_targets = self.total_targets.index((shuffle, ))
        self.total_masks = self.total_masks.index((shuffle, ))

        split_index = int(self.total_data_count * fraction)
        self.train_count = split_index
        self.train_input_data = self.total_inputs[:split_index, :]
        self.train_target_data = self.total_targets[:split_index, :]
        self.train_mask_data = self.total_masks[:split_index, :]

        self.valid_count = self.total_data_count - self.train_count
        self.valid_input_data = self.total_inputs[split_index:, :]
        self.valid_target_data = self.total_targets[split_index:, :]
        self.valid_mask_data = self.total_masks[split_index:, :]

    def setup_batch_size(self, train_batch_size, valid_batch_size):
        """set up batch size for training and validation respectively
        should be called after split_data()
        :parameter train_batch_size: batch size for training samples
        :parameter valid_batch_size: batch size for validation samples
        """
        assert self.train_count % train_batch_size == 0 and self.valid_count % valid_batch_size == 0
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

    def get_train_batch_count(self):
        """compute the number of training batches"""
        return self.train_count // self.train_batch_size

    def get_valid_batch_count(self):
        """compute the number of validation batches"""
        return self.valid_count // self.train_batch_size

    def start_epoch(self):
        """randomly shuffle the training data"""
        shuffle = torch.randperm(self.train_data_count)
        self.train_input_data = self.train_input_data.index((shuffle, ))
        self.train_target_data = self.train_target_data.index((shuffle, ))
        self.train_mask_data = self.train_mask_data.index((shuffle, ))

    def get_train_batch(self, batch_index):
        """get training data according to batch index
        :parameter batch_index: current batch index
        :return: a tuple (inputs, targets, mask)
        """
        start, end = batch_index * self.train_batch_size, (batch_index + 1) * self.train_batch_size
        return self.train_input_data[start:end, :],\
            self.train_target_data[start:end, :],\
            self.train_mask_data[start:end, :]

    def get_valid_batch(self, batch_index):
        """get validation data according to batch index
        :parameter batch_index: current batch index
        :return: a tuple (inputs, targets, mask)
        """
        start, end = batch_index * self.valid_batch_size, (batch_index + 1) * self.valid_batch_size
        return self.valid_input_data[start:end, :],\
            self.valid_target_data[start:end, :],\
            self.valid_mask_data[start:end, :]
