# -*- coding: utf-8 -*-
import torch
from Settings.constants import Players, NodeTypes
from Equity.terminal_equity import TerminalEquity
from Settings.arguments import TexasHoldemAgrument
from Settings.constants import NodeTypes


class PublicTreeCFR:
	def __init__(self, solve_iter=1000, skip_iter=970):
		self.regret_eplision = 10 ** -9
		self._cached_terminal_equities = {}
		self.skip_iter = skip_iter
		self.solve_iter = solve_iter

	# run cfr iter_count times from root node with given start_range
	# @params root the root node
	# @params start_range the start range at root node
	def run_cfr(self, root, start_range):
		root.ranges_absolute = start_range
		for it in range(self.solve_iter):
			self.cfr_iter_dfs(root, it)

	# run cfr depth first search
	def cfr_iter_dfs(self, node, it):
		cp = node.current_player
		op = 1 - cp

		assert (cp == Players.P0 or cp == Players.P1 or cp == Players.CHANCE)

		action_dim, hole_dim = 0, 1

		# store cf values for this node
		values = node.ranges_absolute.clone().fill_(0)

		# [1.0] terminal node, compute v1, v2 = r2U, r1U
		if node.node_type == NodeTypes.TERMINAL_CALL or node.node_type == NodeTypes.TERMINAL_FOLD:
			# get terminal equity, use cached, if not cached, then compute
			key = ' '.join([str(int(x)) for x in node.board])
			terminal_equity = self._cached_terminal_equities.get(key, None)
			if terminal_equity is None:
				terminal_equity = TerminalEquity()
				terminal_equity.set_board(node.board)
				self._cached_terminal_equities[key] = terminal_equity

			# use terminal equity to compute values
			if node.node_type == NodeTypes.TERMINAL_CALL:
				values[0] = torch.matmul(node.ranges_absolute[1], terminal_equity.call_matrix)
				values[1] = torch.matmul(node.ranges_absolute[0], terminal_equity.call_matrix)

			elif node.node_type == NodeTypes.TERMINAL_FOLD:
				values[0] = torch.matmul(node.ranges_absolute[1], terminal_equity.fold_matrix)
				values[1] = torch.matmul(node.ranges_absolute[0], terminal_equity.fold_matrix)
				values[op, :].mul_(-1)
			else:
				raise Exception

			values = values * node.pot
			node.cf_values = values.view_as(node.ranges_absolute)

		# not terminal node
		else:

			action_count = len(node.children)
			current_strategy = None

			if node.current_player == Players.CHANCE:
				print("to do! not implemented yet!")
				raise Exception
			else:
				# compute current strategy for this node

				# init regret and positive regret in first iteration
				if node.regrets is None:
					node.regrets = TexasHoldemAgrument.Tensor(action_count, TexasHoldemAgrument.hole_count)\
						.fill_(self.regret_epsilon)
				if node.positive_regrets is None:
					node.positive_regrets = TexasHoldemAgrument.Tensor(action_count, TexasHoldemAgrument.hole_count)\
						.fill_(self.regret_epsilon)

				# compute positive regrets, use positive regrets to compute current strategy
				node.positive_regrets.copy_(node.regrets)
				node.positive_regrets[torch.le(node.positive_regrets, self.regret_epsilon)] = self.regret_epsilon

				# compute current strategy
				regret_sum = node.positive_regrets.sum(action_dim)
				current_strategy = node.positive_regrets.clone()
				current_strategy.div_(regret_sum.expand_as(current_strategy))

			# end of computing current strategy

			# compute current cfvs [action, players, ranges]
			cf_values_allactions = TexasHoldemAgrument.Tensor(action_count, 2, TexasHoldemAgrument.hole_count)

			children_ranges_absolute = {}

			if node.current_player == Players.CHANCE:
				print("todo!")
				raise Exception
			else:
				ranges_mul_matrix = node.ranges_absolute[cp].repeat(action_count, 1)
				children_ranges_absolute[cp] = torch.mul(current_strategy, ranges_mul_matrix)
				children_ranges_absolute[op] = \
					node.ranges_absolute[op].repeat(action_count, 1).clone()

			for i in range(action_count):
				child_node = node.children[i]
				child_node.ranges_absolute = node.ranges_absolute.clone()

				child_node.ranges_absolute[0].copy_(children_ranges_absolute[0][i])
				child_node.ranges_absolute[1].copy_(children_ranges_absolute[1][i])

				self.cfr_iter_dfs(child_node, it)

				cf_values_allactions[i] = child_node.cf_values

			# use cfvs from actions(children) to compute cfvs for this node
			node.cf_values = TexasHoldemAgrument.Tensor(2, TexasHoldemAgrument.hole_count).fill_(0)

			if cp != Players.CHANCE:
				strategy_mul_matrix = current_strategy.view(action_count, TexasHoldemAgrument.hole_count)
				node.cf_values[cp] = (strategy_mul_matrix * cf_values_allactions[:, cp, :]).sum(0)
				node.cf_values[op] = (cf_values_allactions[:, op, :]).sum(0)
			else:
				raise Exception

			if cp != Players.CHANCE:
				# compute regrets
				current_regrets = \
					cf_values_allactions[:, cp, :].resize_(action_count, TexasHoldemAgrument.hole_count).clone()
				current_regrets.sub_(node.cf_values[cp].view(1, TexasHoldemAgrument.hole_count)
										.expand_as(current_regrets))

				self.update_regrets(node, current_regrets)

				self.update_average_strategy(node, current_strategy, it)

	def update_regrets(self, node, current_regrets):
		# print(current_regrets)
		node.regrets.add_(current_regrets)
		node.regrets[torch.le(node.regrets, self.regret_epsilon)] = self.regret_epsilon

	def update_average_strategy(self, node, current_strategy, it):
		if it > self.skip_iter:

			actions_count = len(node.children)
			if node.strategy is None:
				node.strategy = TexasHoldemAgrument.Tensor(actions_count, TexasHoldemAgrument.hole_count).fill_(0)
			if node.iter_weight_sum is None:
				node.iter_weight_sum = TexasHoldemAgrument.Tensor(TexasHoldemAgrument.hole_count).fill_(0)
			iter_weight_contribution = node.ranges_absolute[node.current_player].clone()
			iter_weight_contribution[torch.le(iter_weight_contribution, 0)] = self.regret_epsilon
			node.iter_weight_sum.add_(iter_weight_contribution)
			iter_weight = iter_weight_contribution / node.iter_weight_sum

			expanded_weight = iter_weight.view(1, TexasHoldemAgrument.hole_count).expand_as(node.strategy)
			old_strategy_scale = 1 - expanded_weight
			node.strategy.mul_(old_strategy_scale)
			node.strategy.add_(current_strategy * expanded_weight)
