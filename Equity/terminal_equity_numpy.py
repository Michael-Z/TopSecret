# -*- coding: utf-8 -*-
import platform
import numpy as np
from Equity.mask import Mask
from ctypes import cdll, c_int
from Tools.card_tools import CardTool
from Settings.arguments import TexasHoldemArgument as Argument

dll = None
if platform.system() == "Windows":
	dll = cdll.LoadLibrary("../so/handeval.dll")
else:
	dll = cdll.LoadLibrary("../so/handeval.so")


class TerminalEquity(object):
	def __init__(self):
		self.hc, self.cc = Argument.hole_count, Argument.card_count
		self.hole_mask = Mask.get_hole_mask()
		self.board_mask = None
		self.call_matrix = None
		self.fold_matrix = None

	def set_board(self, board):
		if len(board) == 2:
			raise NotImplementedError
		if len(board) == 3:
			self.call_matrix = self.compute_flop_call_matrix(current_board=board)
			self.fold_matrix = self.compute_fold_matrix(current_board=board)
		elif len(board) == 4:
			self.call_matrix = self.compute_turn_call_matrix(current_board=board)
			self.fold_matrix = self.compute_fold_matrix(current_board=board)
		elif len(board) == 5:
			view1, view2 = self.construct_strength_view(current_board=board)
			self.call_matrix = self.compute_river_call_matrix(current_board=board, view1=view1, view2=view2)
			self.fold_matrix = self.compute_fold_matrix(current_board=board)
		else:
			raise Exception
		return

	def compute_flop_call_matrix(self, current_board):
		turn_boards = CardTool.get_possible_future_boards(current_board)
		turn_count = len(turn_boards)
		call_matrixs = np.ndarray(shape=(turn_count, self.hc, self.hc), dtype=float)

		for i, turn_board in zip(range(turn_count), turn_boards):
			call_matrixs[i] = self.compute_turn_call_matrix(current_board=turn_board)
		call_matrix = call_matrixs.sum(axis=0, dtype="int32")
		call_matrix = call_matrix.astype(float) / turn_count

		return call_matrix

	def compute_turn_call_matrix(self, current_board):
		river_boards = CardTool.get_possible_future_boards(boards=current_board)
		river_count = len(river_boards)
		call_matrixs = np.ndarray(shape=(river_count, self.hc, self.hc), dtype=float)

		for i, future_board in zip(range(river_count), river_boards):
			view1, view2 = self.construct_strength_view(current_board=future_board)
			call_matrixs[i] = self.compute_river_call_matrix(current_board=future_board, view1=view1, view2=view2)
		call_matrix = call_matrixs.sum(axis=0, dtype="int16")
		call_matrix = call_matrix.astype(float) / river_count

		return call_matrix

	# only river board has strength view, since there are no more future board cards
	def construct_strength_view(self, current_board):
		assert len(current_board) == 5
		# [1.0] compute strength for each hands, -1 indicates impossible hand(conflict with board)
		_strength = (c_int * 1326)()  # param 1
		_board = (c_int * 5)(*current_board)  # param3
		dll.eval5Board(_board, 5, _strength)
		strength = np.array(_strength)  # strength shape is (1326, )

		# set board mask according to strength, shape (1326, )
		board_mask = np.ones(shape=(self.hc, ), dtype="bool")
		board_mask[strength < 0] = False
		assert board_mask.sum() == 1081

		# construct row view and column view, construct win/lose/tie matrix
		# Uij(i for row, j for col) = 1 if hand i < hand j; 0 if hand i == hand j; -1 if hand i > hand j
		strength_view1 = strength.reshape((self.hc, 1)).repeat(repeats=self.hc, axis=1)
		strength_view2 = strength.reshape((1, self.hc)).repeat(repeats=self.hc, axis=0)

		return strength_view1, strength_view2

	# directly construct call matrix with strength view [row view, col view]
	def compute_river_call_matrix(self, current_board, view1, view2):
		call_matrix = np.zeros(shape=(self.hc, self.hc), dtype="bool")

		call_matrix[view1 < view2] = 1  # row < col
		call_matrix[view1 > view2] = -1  # row > col
		call_matrix[view1 == view2] = 0  # row = col, todo advance

		call_matrix[self.hole_mask == False] = 0  # mask out hole cards which conflict each other
		call_matrix[view1 == -1] = 0  # mask out hole card which conflict boards
		call_matrix[view2 == -1] = 0  # mask out hole card which conflict boards

		return call_matrix

	# directly construct fold matrix
	def compute_fold_matrix(self, current_board):
		fold_matrix = np.ones(shape=(self.hc, self.hc), dtype="bool")
		fold_matrix = self.hole_mask.copy()  # make sure player hole don't conflict with opponent hole
		board_mask = Mask.get_board_mask(current_board)  # make sure hole don't conflict with board
		fold_matrix[board_mask == False] = 0

		return fold_matrix

	def get_call_matrix(self):
		return self.call_matrix

	def get_fold_matrix(self):
		return self.fold_matrix
