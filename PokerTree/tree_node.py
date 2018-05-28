

# node_type: inner(include check, chance, player), terminal call/fold
# cp: current player 0, 1
# street: 0, 1, 2, 3
# bets: a list of size 2
# board: a lsit of size[board_card_count]
class Node:
	def __init__(self, street, board, cp, bets, node_type, terminal=False):
		self.street = street
		self.board = board
		self.current_player = cp
		self.bets = bets
		self.pot = min(bets)
		self.node_type = node_type
		self.terminal = terminal

		self.actions = None
		self.children = None
		self.depth = 0

		self.ranges_absolute = None
		self.regrets = None
		self.positive_regrets = None
		self.strategy = None
		self.cf_values = None
		self.iter_weight_sum = None
