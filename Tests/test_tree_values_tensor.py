# -*- coding: utf-8 -*-
import pickle
from PokerTree.tree_values_tensor import TreeValues


root = None
with open("../Data/Tree/root_tensor-1k-2k.dat", "rb") as f:
	root = pickle.load(f)
tv = TreeValues()
tv.compute_values(root)
