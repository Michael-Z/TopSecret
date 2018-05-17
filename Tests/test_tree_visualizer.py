# -*- coding: utf-8 -*-

from PokerTree.tree_builder import TexasHoldemTreeBuilder as TreeBuilder
from Visual.tree_visualizer import TreeVisualizer


tb = TreeBuilder(bet_sizing=None, limit_to_street=False)

# root = tb.build_tree(street=2, initial_bets=[4000, 4000], current_player=0, board=[0, 1, 2, 3])
# file_name = "turn"
# tv = TreeVisualizer()
# tv.graphviz(root, file_name)

root = tb.build_tree(street=3, initial_bets=[4000, 4000], current_player=0, board=[0, 1, 2, 3, 4])
file_name = "river"
tv = TreeVisualizer()
tv.graphviz(root, file_name)


