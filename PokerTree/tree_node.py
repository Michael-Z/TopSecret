

class Node:

	def __init__(self, street, board, current_player, bets, node_type, terminal=False):
		self.street = street
		self.bets = bets.clone()
		self.pot = bets.min()
		self.current_player = current_player
		self.node_type = node_type
		self.board = board
		# self.board_string = None  todo
		self.terminal = terminal
		self.actions = None
		self.depth = 0
		self.children = None
