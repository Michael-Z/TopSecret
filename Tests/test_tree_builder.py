# -*- coding: utf-8 -*-


from PokerTree.tree_builder import TexasHoldemTreeBuilder

tb = TexasHoldemTreeBuilder(bet_sizing=None, limit_to_street=False)
root = tb.build_tree(street=1, initial_bets=[1000, 1000], current_player=0, board=[0, 1, 2])

x = 1
