# -*- coding: utf-8 -*-
from PokerTree.tree_node import Node
from PokerTree.bet_sizing import BetSizing
from Settings.constants import NodeTypes

bs = BetSizing()

node = Node(street=3, board=[1, 2, 3, 4, 5], cp=0, bets=[4000, 12000], node_type=NodeTypes.INNER)
possible_bets = bs.get_possible_bets(node)
assert possible_bets == [[20000, 12000]]
