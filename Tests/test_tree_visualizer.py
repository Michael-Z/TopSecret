# -*- coding: utf-8 -*-
import os
from PokerTree.tree_builder import TexasHoldemTreeBuilder as TreeBuilder
from Visual.tree_visualizer import TreeVisualizer


tb = TreeBuilder(bet_sizing=None, limit_to_street=False)
tv = TreeVisualizer()
file_names = []

root = tb.build_tree(street=2, initial_bets=[4000, 4000], current_player=0, board=[0, 1, 2, 3])
file_name = "turn4k"
tv.graphviz(root, file_name)
file_names.append(file_name)

root = tb.build_tree(street=3, initial_bets=[3000, 3000], current_player=0, board=[0, 1, 2, 3, 4])
file_name = "river3k"
tv.graphviz(root, file_name)
file_names.append(file_name)

root = tb.build_tree(street=3, initial_bets=[4000, 4000], current_player=0, board=[0, 1, 2, 3, 4])
file_name = "river4k"
tv.graphviz(root, file_name)
file_names.append(file_name)


os.chdir("../Data/Visual/")
commands = ["dot %s.dot -Tsvg -o %s.svg" % (file_name, file_name) for file_name in file_names]

for command in commands:
	os.system(command)
