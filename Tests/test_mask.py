# -*- coding: utf-8 -*-

from Equity.mask import Mask

hole_mask = Mask.get_hole_mask()
hole_mask_inverse = Mask.get_hole_mask_inverse()
row_sum = hole_mask.astype(int).sum(1)
col_sum = hole_mask.astype(int).sum(0)
assert (row_sum == 1225).all()
assert (col_sum == 1225).all()
assert ((hole_mask.astype(int) + hole_mask_inverse).astype(int) == 1).all()
for i in range(51):
	for j in range(i + 1, 52):
		index1 = j * (j - 1) // 2 + i
		for m in range(51):
			for n in range(m + 1, 52):
				index2 = n * (n - 1) // 2 + m
				if i == m or i == n or j == m or j == n:
					assert hole_mask[index1, index2] == hole_mask[index2, index1] == False
				else:
					assert hole_mask[index1, index2] == hole_mask[index2, index1] == True


board = [0, 1, 2]
board_mask = Mask.get_board_mask(board=board)
board_mask_inverse = Mask.get_board_mask_inverse(board=board)
assert ((board_mask.astype(int) + board_mask_inverse.astype(int)) == 1).all()
assert len(board_mask) == 1326
assert sum(board_mask) == 1176
for i in range(51):
	for j in range(i + 1, 52):
		index = j * (j - 1) // 2 + i
		if i in board or j in board:
			assert not board_mask[index]
			assert board_mask_inverse[index]
		else:
			assert board_mask[index]
			assert not board_mask_inverse[index]

board = [0, 1, 2, 3]
board_mask = Mask.get_board_mask(board=board)
board_mask_inverse = Mask.get_board_mask_inverse(board=board)
assert len(board_mask) == 1326
assert sum(board_mask) == 1128
for i in range(51):
	for j in range(i + 1, 52):
		index = j * (j - 1) // 2 + i
		if i in board or j in board:
			assert not board_mask[index]
			assert board_mask_inverse[index]
		else:
			assert board_mask[index]
			assert not board_mask_inverse[index]

board = [0, 1, 2, 3, 4]
board_mask = Mask.get_board_mask(board=board)
board_mask_inverse = Mask.get_board_mask_inverse(board=board)
assert len(board_mask) == 1326
assert sum(board_mask) == 1081
for i in range(51):
	for j in range(i + 1, 52):
		index = j * (j - 1) // 2 + i
		if i in board or j in board:
			assert not board_mask[index]
			assert board_mask_inverse[index]
		else:
			assert board_mask[index]
			assert not board_mask_inverse[index]
