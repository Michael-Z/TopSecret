# node_type: inner(include check, chance, player), terminal call/fold
# cp: current player 0, 1
# street: 0, 1, 2, 3
# bets: should be a tensor of size [2]
# board: should be a tensor of size[board_card_count]
class Node:

	def __init__(self, street, board, cp, bets, node_type):
		self.street = street
		self.board = board
		self.current_player = cp
		self.bets = bets
		self.pot = bets.min()
		self.node_type = node_type

		self.actions = None
		self.children = None
		self.depth = 0
