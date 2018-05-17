# -*- coding: utf-8 -*-

from PokerTree.tree_builder import TexasHoldemTreeBuilder as TreeBuilder
from Visual.tree_visualizer import TreeVisualizer


tb = TreeBuilder(bet_sizing=None)
root = tb.build_tree(street=3, initial_bets=[1000, 1000], current_player=0, board=[0, 1, 2, 3, 4])

file_name = "river"
tv = TreeVisualizer()
tv.graphviz(root, file_name)


