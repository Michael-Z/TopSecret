# -*- coding: utf-8 -*-
import platform
import numpy as np
from Equity.mask import Mask
from ctypes import cdll, c_int
from Tools.card_tools import CardTool
from Settings.arguments import TexasHoldemArgument as Argument


class TerminalEquity(object):
	def __init__(self):
		self.dll = cdll.LoadLibrary("../so/hand_eval.dll") if platform.system() == "Windows" else \
					cdll.LoadLibrary("../so/hand_eval.so")
		self.hc, self.cc = Argument.hole_count, Argument.card_count
		self.hole_mask = Mask.get_hole_mask()
		self.board_mask = None
		self.call_matrix = None
		self.fold_matrix = None

	def set_board(self, board):
		call_matrix = self.compute_call_matrix(board=board)
		fold_matrix = self.compute_fold_matrix(board=board)
		self.call_matrix = call_matrix
		self.fold_matrix = fold_matrix

		return

	def compute_call_matrix(self, board):
		board_count = len(board)
		if board_count == 0:
			raise NotImplementedError
		elif board_count == 3:
			call_matrix = self.compute_flop_call_matrix(board=board)
		elif board_count == 4:
			call_matrix = self.compute_turn_call_matrix(board=board)
		elif board_count == 5:
			call_matrix = self.compute_river_call_matrix(board=board)
		else:
			raise Exception

		return call_matrix

	# directly construct fold matrix
	def compute_fold_matrix(self, board):
		fold_matrix = np.ones(shape=(self.hc, self.hc), dtype=float)
		fold_matrix = self.hole_mask.copy()  # make sure player hole don't conflict with opponent hole
		board_mask = Mask.get_board_mask(board)  # make sure hole don't conflict with board
		fold_matrix[board_mask == False] = 0

		return fold_matrix

	def compute_flop_call_matrix(self, board):
		turn_boards = CardTool.get_possible_future_boards(board)
		turn_count = len(turn_boards)
		call_matrixs = np.ndarray(shape=(turn_count, self.hc, self.hc), dtype=float)

		for i, turn_board in zip(range(turn_count), turn_boards):
			call_matrixs[i] = self.compute_turn_call_matrix(board=turn_board)
		call_matrix = call_matrixs.sum(axis=0, dtype=int32)
		call_matrix = call_matrix.astype(float) / turn_count

		return call_matrix

	def compute_turn_call_matrix(self, board):
		river_boards = CardTool.get_possible_future_boards(boards=board)
		river_count = len(river_boards)
		call_matrixs = np.ndarray(shape=(river_count, self.hc, self.hc), dtype=float)

		for i, river_board in zip(range(river_count), river_boards):
			call_matrixs[i] = self.compute_river_call_matrix(board=river_board)
		call_matrix = call_matrixs.sum(axis=0, dtype=float)
		call_matrix = call_matrix.astype(float) / river_count

		return call_matrix

	def compute_river_call_matrix(self, board):
		view1, view2 = self.construct_strength_view(board=board)
		call_matrix = self.compute_final_call_matrix(board=board, view1=view1, view2=view2)

		return call_matrix

	# only river board has strength view, since there are no more future board cards
	def construct_strength_view(self, board):
		assert len(board) == 5
		# [1.0] compute strength for each hands, -1 indicates impossible hand(conflict with board)
		_strength = (c_int * 1326)()  # param 1
		_board = (c_int * 5)(*board)  # param3
		self.dll.eval5Board(_board, 5, _strength)
		strength = np.array(_strength)  # strength shape is (1326, )

		# set board mask according to strength, shape (1326, )
		# board_mask = np.ones(shape=(self.hc, ), dtype="bool")
		# board_mask[strength < 0] = False
		# assert board_mask.sum() == 1081

		# construct row view and column view, construct win/lose/tie matrix
		# Uij(i for row, j for col) = 1 if hand i < hand j; 0 if hand i == hand j; -1 if hand i > hand j
		strength_view1 = strength.reshape((self.hc, 1)).repeat(repeats=self.hc, axis=1)
		strength_view2 = strength.reshape((1, self.hc)).repeat(repeats=self.hc, axis=0)

		return strength_view1, strength_view2

	# directly construct call matrix with strength view [row view, col view]
	def compute_final_call_matrix(self, board, view1, view2):
		call_matrix = np.zeros(shape=(self.hc, self.hc), dtype=float)

		call_matrix[view1 < view2] = 1  # row < col
		call_matrix[view1 > view2] = -1  # row > col
		call_matrix[view1 == view2] = 0  # row = col, todo advance

		call_matrix[self.hole_mask == False] = 0  # mask out hole cards which conflict each other
		call_matrix[view1 == -1] = 0  # mask out hole card which conflict boards
		call_matrix[view2 == -1] = 0  # mask out hole card which conflict boards

		return call_matrix

	def get_call_matrix(self):
		return self.call_matrix

	def get_fold_matrix(self):
		return self.fold_matrix

	# set board must be called first
	# use self.call_matrix and ranges to compute Terminal Call Value
	def compute_call_value(self, ranges):
		values = np.ndarray(shape=(2, Argument.hole_count), dtype=float)
		values[0] = np.matmul(ranges[1], self.call_matrix)
		values[1] = np.matmul(ranges[0], self.call_matrix)

		return values

	# set_board must be called first
	# use self.fold_matrix, ranges and fold_player to compute Terminal Fold Value
	def compute_fold_value(self, ranges, fold_player):
		wp, lp = 1 - fold_player, fold_player  # winer and loser
		values = np.ndarray(shape=(2, Argument.hole_count), dtype=float)
		values[wp] = np.matmul(ranges[lp], self.fold_matrix)
		values[lp] = np.matmul(ranges[wp], -self.fold_matrix)

		return values
