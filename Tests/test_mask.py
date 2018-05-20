# -*- coding: utf-8 -*-

from Equity.mask import Mask

hole_mask = Mask.get_hole_mask()
row_sum = hole_mask.astype("int8").sum(1)
col_sum = hole_mask.astype("int8").sum(0)
assert (row_sum == 1225).all()
assert (col_sum == 1225).all()

board = [0, 1, 2]
board_mask = Mask.get_board_mask(board=board)
assert len(board_mask) == 1326
assert sum(board_mask) == 1176

board = [0, 1, 2, 3]
board_mask = Mask.get_board_mask(board=board)
assert len(board_mask) == 1326
assert sum(board_mask) == 1128

board = [0, 1, 2, 3, 4]
board_mask = Mask.get_board_mask(board=board)
assert len(board_mask) == 1326
assert sum(board_mask) == 1081
