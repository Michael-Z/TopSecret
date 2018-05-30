# -*- coding: utf-8 -*-

from Equity.mask import Mask

hole_mask = Mask.get_hole_mask()
row_sum = hole_mask.float().sum(1)
col_sum = hole_mask.float().sum(0)
assert (row_sum == 1225).all()
assert (col_sum == 1225).all()
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
assert len(board_mask.float()) == 1326
assert sum(board_mask.float()) == 1176
for i in range(51):
    for j in range(i + 1, 52):
        index = j * (j - 1) // 2 + i
        if i in board or j in board:
            assert not board_mask[index]
        else:
            assert board_mask[index]

board = [0, 1, 2, 3]
board_mask = Mask.get_board_mask(board=board)
assert len(board_mask.float()) == 1326
assert sum(board_mask.float()) == 1128
for i in range(51):
    for j in range(i + 1, 52):
        index = j * (j - 1) // 2 + i
        if i in board or j in board:
            assert not board_mask[index]
        else:
            assert board_mask[index]

board = [0, 1, 2, 3, 4]
board_mask = Mask.get_board_mask(board=board)
assert len(board_mask.float()) == 1326
assert sum(board_mask.float()) == 1081
for i in range(51):
    for j in range(i + 1, 52):
        index = j * (j - 1) // 2 + i
        if i in board or j in board:
            assert not board_mask[index]
        else:
            assert board_mask[index]
