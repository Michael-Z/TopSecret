# -*- coding: utf-8 -*-
import torch
import platform
from ctypes import cdll, c_int
from Equity.mask import Mask
from Settings.arguments import TexasHoldemAgrument

dll = None
if platform.system() == "Windows":
	dll = cdll.LoadLibrary("../so/handeval.dll")
else:
	dll = cdll.LoadLibrary("./so/handeval.so")


class TerminalEquity(object):
	def __init__(self):
		self.hole_mask = TexasHoldemAgrument.Tensor(Mask.get_hole_mask())
		self.board_mask = None
		self.call_matrix = None
		self.fold_matrix = None

	def set_board(self, board):
		# matrix [1326*1326]

		# [1.0] set call matrix (only works for last round)
		assert board.size(0) == 5

		call_matrix = TexasHoldemAgrument.Tensor(TexasHoldemAgrument.hole_count, TexasHoldemAgrument.hole_count)\
			.fill_(0)
		# self.board_mask = Mask.get_board_mask(board)

		# hand evaluation, get strength vector
		_strength = (c_int * 1326)()
		_board = (c_int * 5)()
		for i in range(board.size(0)):
			_board[i] = int(board[i])
		# strength indicates how large the hand is, -1 indicates impossible hand(conflict with board)
		dll.eval5Board(_board, 5, _strength)
		strength_list = [x for x in _strength]
		strength = TexasHoldemAgrument.Tensor(strength_list)

		# set board mask according to strength
		self.board_mask = strength.clone().fill_(1)
		self.board_mask[strength < 0] = 0

		assert int((self.board_mask > 0).sum()) == 1081

		# construct row view and column view, construct win/lose/tie matrix
		# Uij(i for row, j for col) = 1 if hand i < hand j; 0 if hand i == hand j; -1 if hand i > hand j
		strength_view1 = strength.view(TexasHoldemAgrument.hole_count, 1).expand_as(call_matrix)
		strength_view2 = strength.view(1, TexasHoldemAgrument.hole_count).expand_as(call_matrix)

		call_matrix[torch.lt(strength_view1, strength_view2)] = 1
		call_matrix[torch.gt(strength_view1, strength_view2)] = -1
		# call_matrix[torch.eq(strength_view1, strength_view2)] = 0
		# mask out hole cards which conflict each other
		call_matrix[self.hole_mask < 1] = 0
		# mask out hole card which conflict boards
		call_matrix[strength_view1 == -1] = 0
		call_matrix[strength_view2 == -1] = 0

		# [2.0] set fold matrix
		fold_matrix = TexasHoldemAgrument.Tensor(TexasHoldemAgrument.hole_count, TexasHoldemAgrument.hole_count)
		# make sure player hole don't conflict with opponent hole
		fold_matrix.copy_(self.hole_mask)
		# make sure hole don't conflict with board
		fold_matrix[strength_view1 == -1] = 0
		fold_matrix[strength_view2 == -1] = 0

		self.call_matrix = call_matrix
		self.fold_matrix = fold_matrix
