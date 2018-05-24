# -*- coding: utf-8 -*-
import platform
import numpy as np
from random import shuffle
from Equity.mask import Mask
from ctypes import cdll, c_int
from Equity.terminal_equity_numpy import TerminalEquity

dll = cdll.LoadLibrary("../so/hand_eval.dll") if platform.system() == "Windows" \
	else cdll.LoadLibrary("../so/hand_eval.so")

deck = list(range(52))


def test_terminal_equity_numpy():
	shuffle(deck)
	board = deck[0:5]
	te = TerminalEquity()
	te.set_board(board=board)
	call_matrix = te.get_call_matrix()
	fold_matrix = te.get_fold_matrix()
	hole_mask = Mask.get_hole_mask()

	# eval using dll
	_strength = (c_int * 1326)()  # param 1
	_board = (c_int * 5)(*board)  # param3
	dll.eval5Board(_board, 5, _strength)
	strenght_list = np.array(_strength, dtype=int)

	for i in range(1325):
		for j in range(i + 1, 1326):
			assert call_matrix[i, j] == -call_matrix[j, i]

	# test call matrix
	for i in range(1325):
		for j in range(i + 1, 1326):
			if strenght_list[i] == -1 or strenght_list[j] == -1:
				# call_matrix[i, j] should be 0, because is conflict with board
				assert call_matrix[i, j] == 0
			elif not hole_mask[i][j]:
				assert call_matrix[i, j] == 0
			elif strenght_list[i] > strenght_list[j]:
				try:
					assert call_matrix[i, j] == -1
				except AssertionError:
					print(call_matrix[i, j], strenght_list[i], strenght_list[j])
					raise Exception
			elif strenght_list[i] == strenght_list[j]:
				assert call_matrix[i, j] == 0
			else:
				assert call_matrix[i, j] == 1

	# test fold matrix
	for i in range(51):
		for j in range(i + 1, 52):
			index1 = j * (j - 1) // 2 + i
			for m in range(51):
				for n in range(m + 1, 52):
					index2 = n * (n - 1) // 2 + m
				if i == m or i == n or j == m or j == n or i in board or j in board or m in board or n in board:
					assert fold_matrix[index1, index2] == fold_matrix[index2, index1] == 0
				else:
					assert fold_matrix[index1, index2] == fold_matrix[index2, index1] == 1

for t in range(100):
	test_terminal_equity_numpy()

