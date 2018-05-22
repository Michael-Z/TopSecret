# -*- coding: utf-8 -*-
import os
import pickle
import random
import numpy as np
from random import shuffle
from Equity.mask import Mask
from Settings.constants import Players
from Range.ehs import ExpectedHandStrength
from CFR.public_cfr_numpy import PublicTreeCFR
from Range.range_generator import RangeGenerator
from Settings.arguments import TexasHoldemArgument as Arguments
from PokerTree.tree_builder import TexasHoldemTreeBuilder as TreeBuilder
from HandIsomorphism.hand_isomorphism_encapsulation import HandIsomorphismEncapsulation as HandIsomorphism


class DataGenerator:
	def __init__(self):
		self.save_folder = "../Data/TrainingSamples/Texas/CardValue/"
		self.round = None  # current round [1, 2, 3], shouldn't generate PREFLOP data
		self.board_count = None  # how many board cards till current round
		self.save_path = None
		self.solver = None
		self.range_generator = None
		self.tree_builder = None

		# self.batch_size = Arguments.batch_size
		self.batch_size = 10
		self.batch_count = None  # how many batches

		self.deck = None
		self.intervals = [(100, 100), (200, 400), (400, 2000), (2000, 6000), (6000, 19950)]

		self.ehs = ExpectedHandStrength(file_path="../Data/EHS/")
		self.hand_iso = HandIsomorphism()

		self.inputs_ph = np.ndarray(shape=(self.batch_size, 2 * 1326 + 1), dtype=float)  # place holder
		self.targets_ph = np.ndarray(shape=(self.batch_size, 2 * 1326), dtype=float)  # place holder
		self.mask_ph = np.ndarray(shape=(self.batch_size, 1326), dtype=float)  # place holder

		self.counter = len(os.listdir(self.save_folder)) // 3

	# @param filename
	# @param rd should be 2, 3, 4
	def setup(self, rd, solve_iter=1000, skip_iter=970):
		self.round = rd
		self.hand_iso.setup_by_round(rounds=rd + 1, mode="board")
		self.board_count = [0, 3, 4, 5][rd]
		self.solver = PublicTreeCFR(hand_iso=self.hand_iso, solve_iter=solve_iter, skip_iter=skip_iter)
		self.range_generator = RangeGenerator(ehs=self.ehs)
		self.tree_builder = TreeBuilder(bet_sizing=None, limit_to_street=True)
		self.deck = list(range(52))

	def generate_data(self, data_count):
		if self.round == 0:
			raise Exception
		if self.round == 1:
			raise NotImplementedError
		elif self.round == 2:
			raise NotImplementedError
		elif self.round == 3:
			self.generate_river_data(data_count=data_count)

	def generate_river_data(self, data_count):
		self.batch_count = data_count // self.batch_size

		for i in range(self.batch_count):
			# [1.0] generate random poker situations
			# board list[b0, b1, ...], ranges ndarray[2 * batch_size * 1326], pots ndarray[batch_size,]
			board, ranges, pots = self.generate_random_poker_situation()
			pots = pots / Arguments.stack  # normalize
			mask = Mask.get_board_mask(board)

			# copy mask to mask place holder
			mask = mask.reshape((1, Arguments.hole_count))
			self.mask_ph = mask.repeat(repeats=self.batch_size, axis=0)
			# copy pots to input place holder
			self.inputs_ph[:, -1] = pots

			# [2.0] solve random poker situations
			for j in range(self.batch_size):
				bets = [pots[j]] * 2
				root = self.tree_builder.build_tree(street=self.round, initial_bets=bets,
													current_player=Players.P0, board=board)
				start_range = ranges[:, j, :]  # shape (2,1326)
				assert start_range.shape == (2, 1326)
				self.solver.run_cfr(root, start_range=start_range)
				cf_values = root.cf_values
				# inputs, pots already handled before
				self.inputs_ph[j, :1326] = ranges[0, j, :]
				self.inputs_ph[j, 1326:2652] = ranges[1, j, :]
				# targets
				self.targets_ph[j, :1326] = cf_values[0]
				self.targets_ph[j, 1326:] = cf_values[1]
				# mask, already handled before loop
			# end for
			# save a batch of data
			self.save_batch_data()

	def save_batch_data(self):
		inputs_save_path = "%s.%05d.batch_size_%d.inputs" % (self.save_folder, count, self.batch_size)
		targets_save_path = "%s.%05d.batch_size_%d.targets" % (self.save_folder, count, self.batch_size)
		mask_save_path = "%s.%05d.batch_size_%d.mask" % (self.save_folder, count, self.batch_size)
		with open(inputs_save_path, "wb") as f:
			pickle.dump(self.inputs_ph, f)
		with open(targets_save_path, "wb") as f:
			pickle.dump(self.targets_ph, f)
		with open(mask_save_path, "wb") as f:
			pickle.dump(self.mask_ph, f)
		self.counter += 1

		return

	# @return board, ranges, pots. [one board, a batch of ranges and pots]
	# board -> list, ranges -> ndarray(2 * batch_size * 1326), pots -> ndarray(batch_size, )
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
		pots = np.ndarray(shape=(self.batch_size, ), dtype=float)
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
	dg.generate_data(data_count=100)
	e = time.time()
	print(e - s)

if __name__ == '__main__':
	main()
