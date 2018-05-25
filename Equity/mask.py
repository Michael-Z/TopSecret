# -*- coding: utf-8 -*-
import torch
import numpy as np
from Settings.arguments import TexasHoldemArgument as Argument


# all return value are bool ndarray
class Mask:
	#  mask out conflict hole combinations
	hole_mask = None  # FloatTensor (1326, 1326), compute once

	# @return FloatTensor (1326, 1326)
	@classmethod
	def get_hole_mask(cls):
		if cls.hole_mask is None:
			valid_hole_mask = np.ones(shape=(1326, 1326), dtype="bool")

			# p1 hole:(s_card, b_card), p2 hole(s, b), p1, p2 doesn't share same card
			for s_card in range(51):
				for b_card in range(s_card + 1, 52):
					row_index = b_card * (b_card - 1) // 2 + s_card

					# find conflict col_index
					for s in range(51):
						for b in range(s + 1, 52):
							if s == s_card or b == b_card or s == b_card or b == s_card:
								col_index = b * (b - 1) // 2 + s
								valid_hole_mask[row_index][col_index] = False
			cls.hole_mask = valid_hole_mask
		return cls.hole_mask

	# @param board should be a list
	# @return one dim bool ndarray, shape is (1326, )
	@classmethod
	def get_board_mask(cls, board):
		s = set(board)
		out = np.ones(shape=(1326, ), dtype="bool")
		for s_card in range(51):
			for b_card in range(s_card + 1, 52):
				if s_card in s or b_card in s:
					index = b_card * (b_card - 1) // 2 + s_card
					out[index] = False
		return out
