# -*- coding: utf-8 -*-
import time
import pickle
import numpy as np
from Range.ehs import ExpectedHandStrength
from CFR.public_cfr_numpy import PublicTreeCFR
from Range.range_generator import RangeGenerator
from HandIsomorphism.hand_isomorphism_encapsulation import HandIsomorphismEncapsulation as HandIsomorphism
from PokerTree.tree_builder import TexasHoldemTreeBuilder as TreeBuilder


board = [0, 32, 40, 44, 48]
rg = RangeGenerator(ehs=ExpectedHandStrength(file_path="../Data/EHS/"))
rg.set_board(board=board)
ranges_0 = rg.generate_ranges(1)
ranges_1 = rg.generate_ranges(1)

start_range = np.ndarray(shape=(2, 1326), dtype=float)
start_range[0] = ranges_0[0]
start_range[1] = ranges_1[0]

street = 3
tb = TreeBuilder(bet_sizing=None)
root = tb.build_tree(street=street, initial_bets=[2000, 2000], current_player=0, board=board)
hand_iso = HandIsomorphism()
hand_iso.setup_by_round(rounds=street + 1, mode="board")
solver = PublicTreeCFR(hand_iso=hand_iso)
s = time.time()
solver.run_cfr(root, start_range)
e = time.time()
print(e - s)

with open("../Data/Tree/root.dat", "wb") as f:
	pickle.dump(root, f)
