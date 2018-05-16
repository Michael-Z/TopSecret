# -*- coding: utf-8 -*-
import torch
import random
import numpy as np
from random import shuffle
from Equity.mask import Mask
from Settings.constants import Players
from Range.ehs import ExpectedHandStrength
from CFR.public_tree_cfr import PublicTreeCFR
from Range.range_generator import RangeGenerator
from Settings.arguments import TexasHoldemAgrument as Arguments
from PokerTree.tree_builder import TexasHoldemTreeBuilder as TreeBuilder


class DataGenerator:
	def __init__(self):
		self.save_folder = "../Data/TrainingSamples/Texas/"
		self.round = None
		self.board_count = None
		self.save_path = None
		self.solver = None
		self.range_generator = None
		self.tree_builder = None
		self.train_suffix = ".train"
		self.valid_suffix = ".valid"

		self.train_batch_count = None
		self.valid_batch_count = None
		self.deck = None

		self.batch_size = 10
		# self.batch_size = Arguments.batch_size
		self.intervals = [(100, 100), (200, 400), (400, 2000), (2000, 6000), (6000, 19950)]

		self.inputs_ph = None  		# placeholder of inputs
		self.targets_ph = None		# placeholder of targets
		self.mask_ph = None			# placeholder of mask

		self.ehs = ExpectedHandStrength(file_path="../Data/EHS/")

	# @param filename
	# @param rd should be 2, 3, 4
	def setup(self, rd, solve_iter=1000, skip_iter=970):
		self.round = rd
		self.board_count = [0, 3, 4, 5][rd]
		self.solver = PublicTreeCFR(solve_iter=solve_iter, skip_iter=skip_iter)
		self.range_generator = RangeGenerator(ehs=self.ehs)
		self.tree_builder = TreeBuilder(bet_sizing=None, limit_to_street=True)
		self.deck = list(range(52))

	def generate_data(self, train_count, valid_count):
		self.train_batch_count = train_count // self.batch_size
		self.generate_train_data(data_count=train_count)

		# self.valid_batch_count = valid_count // self.batch_size
		# self.generate_valid_data(data_count=valid_count)

	def generate_train_data(self, data_count):
		# initialize inputs, targets, masks' place holder
		if self.inputs_ph is None:
			self.inputs_ph = Arguments.Tensor(self.batch_size, 2 * 1326 + 1)
			self.targets_ph = Arguments.Tensor(self.batch_size, 2 * 1326)
			self.mask_ph = Arguments.Tensor(self.batch_size, 1326)

		for i in range(self.train_batch_count):
			# [1.0] generate random poker situations
			# board [b0, b1, ...], ranges [2 * batch_size * 1326], pots[batch_size * 1]
			board, ranges, pots = self.generate_random_poker_situation()
			ranges_tensor = Arguments.Tensor(ranges)
			pots_tensor = Arguments.Tensor(pots).div_(Arguments.stack)
			mask = Mask.get_board_mask(board)
			mask_tensor = Arguments.Tensor(mask)
			# mask
			self.mask_ph.copy_(mask_tensor.expand_as(self.mask_ph))
			# [2.0] solve random poker situations
			for j in range(self.batch_size):
				bets = [pots[j]] * 2
				root = self.tree_builder.build_tree(street=self.round, initial_bets=bets,
													current_player=Players.P0, board=board)
				start_range = ranges_tensor[:, j, :]
				self.solver.run_cfr(root, start_range=start_range)
				cf_values = root.cf_values
				# inputs
				self.inputs_ph[j, :1326].copy_(ranges_tensor[0, j, :])
				self.inputs_ph[j, 1326:2652].copy_(ranges_tensor[1, j, :])
				self.inputs_ph[j, -1] = pots_tensor[j]
				# targets
				self.targets_ph[j, :1326].copy_(cf_values[0])
				self.targets_ph[j, 1326:].copy_(cf_values[1])
				# mask already handled before loop
			# end for
			# save a batch of data
			self.save_train_data(i)

	def save_train_data(self, count):
		inputs_save_path = "%s.%05d.train.inputs" % (self.save_folder, count)
		targets_save_path = "%s.%05d.train.targets" % (self.save_folder, count)
		mask_save_path = "%s.%05d.train.mask" % (self.save_folder, count)
		torch.save(self.inputs_ph.float(), inputs_save_path)
		torch.save(self.targets_ph.float(), targets_save_path)
		torch.save(self.mask_ph.float(), mask_save_path)

	def generate_valid_data(self, data_count):
		pass

	# @return board, ranges, pots. [one board, a batch of ranges and pots]
	# board should be a list, ranges ndarray(2 * batch_size * 1326), pots ndarray(batch_size * 1)
	def generate_random_poker_situation(self):
		# [1.0] generate board cards
		shuffle(self.deck)
		board = self.deck[:self.board_count]

		# [2.0] generate ranges according to board cards
		ranges = np.ndarray(shape=(2, self.batch_size, 1326), dtype=float)
		self.range_generator.set_board(board=board)
		ranges[0] = self.range_generator.generate_ranges(batch_size=self.batch_size)
		ranges[1] = self.range_generator.generate_ranges(batch_size=self.batch_size)

		# [3.0] generate pots
		# [100, 100), [200, 400), [400, 2000), [2000, 6000), [6000, 19950)
		pots = np.ndarray(shape=(self.batch_size, 1), dtype="int16")
		# [3.1] select uniformly from [0-4]
		for i in range(self.batch_size):
			left, right = interval = self.intervals[random.randint(0, len(self.intervals) - 1)]
			if left == right:
				pots[i] = left
			else:
				pots[i] = random.randint(left, right - 1)
		return board, ranges, pots


def main():
	dg = DataGenerator()
	dg.setup(rd=3)
	import time
	s = time.time()
	dg.generate_data(train_count=10, valid_count=100)
	e = time.time()
	print(e - s)

if __name__ == '__main__':
	main()
