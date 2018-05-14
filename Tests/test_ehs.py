# -*- coding: utf-8 -*-
from Range.ehs import ExpectedHandStrength

ehs = ExpectedHandStrength(file_path="../Data/EHS/")

assert len(ehs.preflop_ehs) == 169
assert len(ehs.flop_ehs) == 1286792
assert len(ehs.turn_ehs) == 13960050
assert len(ehs.river_ehs) == 123156254

board = [0, 1, 2]
rd = 1
ehs_list = ehs.get_possible_hand_ehs(board_cards=board, rd=rd)
assert len(ehs_list) == 1326
for e in ehs_list:
	assert e == -1 or 0 <= e <= 1
