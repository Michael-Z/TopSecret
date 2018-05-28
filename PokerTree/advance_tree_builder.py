# -*- coding: utf-8 -*-
import torch
from PokerTree.tree_node import Node
from Tools.card_tools import CardTool
from PokerTree.bet_sizing import BetSizing
from Settings.constants import Actions, Players, NodeTypes
from Settings.arguments import TexasHoldemArgument as Arguments


class AdvanceTreeBuilder:
	def __init__(self, bet_sizing=None, limit_to_street=True):
		self.bet_sizing = bet_sizing or BetSizing(pot_fractions=[1])
		self.limit_to_street = limit_to_street
		self.preflop_bet_sizing = [BetSizing(pot_fractions=[0.5, 1]), BetSizing(pot_fractions=[0.5, 1, 2])]
		self.flop_bet_sizing = [BetSizing(pot_fractions=[0.5, 1]), BetSizing(pot_fractions=[1])]
		self.turn_bet_sizing = [BetSizing(pot_fractions=[0.5, 1]), BetSizing(pot_fractions=[1])]
		self.river_bet_sizing = [BetSizing(pot_fractions=[0.5, 1, 2]), BetSizing(pot_fractions=[0.5, 1, 2])]
		self.all_round_bet_sizing = (self.preflop_bet_sizing, self.flop_bet_sizing,
									self.turn_bet_sizing, self.river_bet_sizing)

	def build_tree(self, street, initial_bets, current_player, board):
		root = Node(street, board, current_player, initial_bets, node_type=NodeTypes.INNER)
		root.street_player_action_count = 0  # how many action made in current street
		self._build_tree_dfs(root)
		return root

	def _build_tree_dfs(self, node):
		children = self._get_children(node)
		node.children = children
		child_count = len(children)
		node.actions = torch.FloatTensor(child_count)
		depth = 0
		for i in range(child_count):
			children[i].parent = node
			self._build_tree_dfs(children[i])
			depth = max(depth, children[i].depth)
			if i == 0:
				node.actions[i] = Actions.FOLD
			elif i == 1:
				node.actions[i] = Actions.CCALL
			else:
				node.actions[i] = max(children[i].bets)
		node.depth = depth + 1
		return node

	def _get_children_node(self, node):
		children = []
		street, board = node.street, node.board
		cp, op = node.current_player, 1 - node.current_player

		if cp in (Players.P0, Players.P1):  # [1.0] current node is player node
			# [1.1] FOLD is always valid
			fold_node = Node(street, board, op, node.bets, NodeTypes.TERMINAL_FOLD, True)
			fold_node.street_player_action_count = node.street_action_count + 1
			children.append(fold_node)

			# [1.2] CALL, CHECK; TRANSITION CALL; TERMINAL CALL
			# if first action of a round is CHECK/CALL, it won't lead to round transition
			if node.street_action_count == 0:
				ccall_node = Node(street, board, op, node.bets, NodeTypes.CHECK)
				ccall_node.street_player_action_count = node.street_action_count + 1
				children.append(ccall_node)
			# TRANSITION CALL' street must be in [0, 1, 2], and opponent bet must < stack(20000)
			elif street <= 2 and node.bets[op] < Arguments.stack:
				bets = [max(node.bets)] * 2
				transition_call_node = Node(street, board, Players.CHANCE, bets, NodeTypes.CHANCE)
				transition_call_node.street_player_action_count = 0
				children.append(transition_call_node)
			else:
				assert street == 3 or node.bets[op] == Arguments.stack
				bets = [max(node.bets)] * 2
				terminal_call_node = Node(node.street, node.board, op, bets, NodeTypes.TERMINAL_CALL, True)
				terminal_call_node.street_player_action_count = node.street_action_count + 1
				children.append(terminal_call_node)

			# [1.3] raise actions (including ALLIN)
			bet_sizing = self.bet_sizing
			if node.street_player_action_count < 2:
				bet_sizing = self.all_round_bet_sizing[street][node.street_player_action_count]

			possible_bets = self.bet_sizing.get_possible_bets(node)
			for bets in possible_bets:
				child_node = Node(node.street, node.board, op, bets, NodeTypes.INNER)
				child_node.street_player_action_count = node.street_player_action_count + 1
				children.append(child_node)
		elif cp == Players.CHANCE:  # [2.0] current node is chance node
			children = []
			assert node.current_player == Players.CHANCE
			if self.limit_to_street:
				return children

			next_street = node.street + 1
			next_boards = self.get_boards(node.street + 1, node.board)  # get all possible future boards
			for i, next_board in zip(range(len(next_boards)), next_boards):
				child_node = Node(next_street, next_board, Players.P0, node.bets, NodeTypes.INNER)
				# chance action won't affect player action counter
				children.append(child_node)
			return children
		else:
			raise Exception
		return children

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
