import torch
from PokerTree.tree_node import Node
from Settings.constants import Actions, Players, NodeTypes
from Tools.card_tools import CardTool
from PokerTree.bet_sizing import BetSizing


class TexasHoldemTreeBuilder:

	def __init__(self, bet_sizing, limit_to_street=True):
		self.limit_to_street = limit_to_street
		self.bet_sizing = bet_sizing or BetSizing(pot_fractions=[1])

	def build_tree(self, street, initial_bets, current_player, board):
		root = Node(street, board, current_player, initial_bets, node_type="inner", terminal=False)
		self._build_tree_dfs(root)
		return root

	def _build_tree_dfs(self, node):
		children = self._get_children(node)
		node.children = children
		child_count = len(children)
		node.actions = [None] * child_count
		depth = 0
		for i in range(child_count):
			children[i].parent = node
			self._build_tree_dfs(children[i])
			node.depth = max(depth, children[i].depth)
			if i == 0:
				node.actions[i] = Actions.FOLD
			elif i == 1:
				node.actions[i] = Actions.CCALL
			else:
				node.actions[i] = children[i].bets.max()
		node.depth += 1
		return node

	# create children nodes after parent node
	# players at children node might be Players.CHANCE or P0 or P1
	def _get_children(self, node):
		if node.terminal:
			return []
		elif node.current_player == Players.CHANCE:
			return self._get_children_of_chance(node)
		else:
			return self._get_children_of_player(node)

	# create children nodes after player node
	def _get_children_of_player(self, node):
		assert node.current_player != Players.CHANCE

		children = []

		# [1.0] fold is always valid
		fold_node = Node(node.street, node.board, 1 - node.current_player, node.bets, NodeTypes.TERMINAL_FOLD, True)
		children.append(fold_node)

		# check flag
		check = node.street > 0 and node.bets[0] == node.bets[1] and node.current_player == Players.P0

		# transition flag
		cc = node.bets[node.current_player] == node.bets[1 - node.current_player] and node.current_player == Players.P1
		rc = node.bets[node.current_player] < node.bets[1 - node.current_player]
		transition = 0 < node.street < 3 and (cc or rc)
		if check:			# [2.0] check, bets equal, not preflop, cp == P0
			check_node = Node(node.street, node.board, 1 - node.current_player, node.bets, NodeTypes.CHECK, False)
			children.append(check_node)
		elif transition:	# [3.0] transition, cc or *rc
			chance_node = Node(node.street, node.board, Players.CHANCE, node.bets, NodeTypes.CHANCE, False)
			children.append(chance_node)
		else:				# [4.0] terminal call
			bets = node.bets.clone().fill_(node.bets.max())
			terminal_call_node = Node(node.street, node.board, Players.NONE, bets, NodeTypes.TERMINAL_CALL, True)
			children.append(terminal_call_node)
		#  [5.0] raise actions
		possible_bets = self.bet_sizing.get_possible_bets(node)
		for bet in possible_bets:
			bet_tensor = torch.FloatTensor(bet)
			child_node = Node(node.street, node.board, 1 - node.current_player, bet_tensor, NodeTypes.INNER, False)
			children.append(child_node)
		return children

	def _get_children_of_chance(self, node):
		children = []
		assert node.current_player == Players.CHANCE
		if self.limit_to_street:
			return children

		next_boards = self.get_boards(node.street + 1, node.board)
		for i, next_board in zip(range(len(next_boards)), next_boards):
			board_tensor = torch.FloatTensor(next_board)
			child_node = Node(node.street + 1, board_tensor, Players.P0, node.bets, NodeTypes.INNER, False)
			children.append(child_node)
		return children

	# return a list contains every possible boards given previous board
	# @param street current round number
	# @param old_board board cards of last round, should be torch.FloatTensor
	def get_boards(self, street, old_board):
		new_boards = []
		if street < 2:
			raise NotImplementedError
		# TURN
		elif street == 2:
			new_boards = CardTool.get_possible_future_boards(old_board)
			assert len(new_boards) == 52 - 3
		# RIVER
		elif street == 3:
			new_boards = CardTool.get_possible_future_boards(old_board)
			assert len(new_boards) == 52 - 4
		else:
			raise Exception
		return new_boards
