from PokerTree.tree_node import Node
from Settings.constants import Actions, Players, NodeTypes


class TexasHoldemTreeBuilder:

	def __init__(self, limit_to_street, bet_sizing):
		self.limit_to_street = limit_to_street
		self.bet_sizing = bet_sizing

	def build_tree(self, street, initial_bets, current_player, board):
		root = Node(street, board, current_player, initial_bets, node_type="inner", terminal=False)
		self._build_tree_dfs(root)
		return root

	def _build_tree_dfs(self, node):
		children = self._get_children(node)
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
			return self._get_children_of_chance()
		else:
			return self._get_children_of_player()

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
		return children

	def _get_children_of_chance(self, node):
		pass
