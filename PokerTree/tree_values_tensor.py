# -*- coding: utf-8 -*-
import torch
from Settings.constants import Players, NodeTypes
from Equity.terminal_equity_numpy import TerminalEquity
from Settings.arguments import TexasHoldemArgument as Argument


class TreeValues:
	# Constructor
	def __init__(self):
		self.terminal_equity = TerminalEquity()

	# Recursively calculate the counterfactual values for each player at each
	# node of the tree using the saved strategy profile.
	#
	# The cfvs for each player in the given strategy profile when playing against
	# each other is stored in the `cf_values` field for each node. The cfvs for
	# a best response against each player in the profile are stored in the
	# `cf_values_br` field for each node.
	# @param node the current node
	# @local
	def _compute_values_dfs(self, node):
		cp, op = node.current_player, 1 - node.current_player
		# compute values using terminal_equity in terminal nodes
		if node.node_type in (NodeTypes.TERMINAL_CALL, NodeTypes.TERMINAL_FOLD):
			self.terminal_equity.set_board(node.board)
			values = node.ranges_absolute.clone().fill_(0)

			if node.node_type == NodeTypes.TERMINAL_FOLD:
				values[cp] = torch.matmul(node.ranges_absolute[op], self.terminal_equity.fold_matrix_tensor)
				values[op] = torch.matmul(node.ranges_absolute[cp], -self.terminal_equity.fold_matrix_tensor)
			elif node.node_type == NodeTypes.TERMINAL_CALL:
				values[cp] = torch.matmul(node.ranges_absolute[op], self.terminal_equity.call_matrix_tensor)
				values[op] = torch.matmul(node.ranges_absolute[cp], self.terminal_equity.call_matrix_tensor)

			# multiply by the pot
			values = values * node.pot

			node.cf_values = values.view_as(node.ranges_absolute)
			node.cf_values_br = values.view_as(node.ranges_absolute)
		else:

			actions_count = len(node.children)
			ranges_size = node.ranges_absolute.size(1)

			# [[actions, players, ranges]]
			cf_values_allactions = Argument.Tensor(len(node.children), 2, ranges_size).fill_(0)
			cf_values_br_allactions = Argument.Tensor(len(node.children), 2, ranges_size).fill_(0)

			for i in range(len(node.children)):
				child_node = node.children[i]
				self._compute_values_dfs(child_node)
				cf_values_allactions[i] = child_node.cf_values
				cf_values_br_allactions[i] = child_node.cf_values_br

			node.cf_values = Argument.Tensor(2, ranges_size).fill_(0)
			node.cf_values_br = Argument.Tensor(2, ranges_size).fill_(0)

			# strategy = [[actions x range]]
			#        strategy_mul_matrix = node.strategy.view(actions_count, ranges_size)
			strategy_mul_matrix = node.strategy.view(-1, ranges_size)

			# compute CFVs given the current strategy for this node
			if node.current_player == Players.CHANCE:
				node.cf_values = cf_values_allactions.sum(0)[0]
				node.cf_values_br = cf_values_br_allactions.sum(0)[0]
			else:
				node.cf_values[node.current_player] = torch.mul(strategy_mul_matrix,
																cf_values_allactions[:, node.current_player, :]).sum(0)
				node.cf_values[1 - node.current_player] = (cf_values_allactions[:, 1 - node.current_player, :]).sum(0)

				# compute CFVs given the BR strategy for this node
				node.cf_values_br[1 - node.current_player] = cf_values_br_allactions[:, 1 - node.current_player, :].sum(
					0)
				node.cf_values_br[node.current_player] = cf_values_br_allactions[:, node.current_player, :].max(0)[0]

		# counterfactual values weighted by the reach prob
		node.cfv_infset = Argument.Tensor(2)
		node.cfv_infset[0] = (node.cf_values[0] * node.ranges_absolute[0]).sum()
		node.cfv_infset[1] = (node.cf_values[1] * node.ranges_absolute[1]).sum()

		# compute CFV-BR values weighted by the reach prob
		node.cfv_br_infset = Argument.Tensor(2)
		node.cfv_br_infset[0] = (node.cf_values_br[0] * node.ranges_absolute[0]).sum()
		node.cfv_br_infset[1] = (node.cf_values_br[1] * node.ranges_absolute[1]).sum()

		node.epsilon = node.cfv_br_infset - node.cfv_infset
		node.exploitability = node.epsilon.mean()

	# Compute the self play and best response values of a strategy profile on
	# the given game tree.
	#
	# The cfvs for each player in the given strategy profile when playing against
	# each other is stored in the `cf_values` field for each node. The cfvs for
	# a best response against each player in the profile are stored in the
	# `cf_values_br` field for each node.
	#
	# @param root The root of the game tree. Each node of the tree is assumed to
	# have a strategy saved in the `strategy` field.
	# @param[opt] starting_ranges probability vectors over player private hands
	# at the root node (default uniform)
	def compute_values(self, root):
		self._compute_values_dfs(root)
		print(root.exploitability)
