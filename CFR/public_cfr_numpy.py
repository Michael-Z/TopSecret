import numpy as np
from Settings.constants import Players, NodeTypes
from Equity.terminal_equity_numpy import TerminalEquity
from Settings.arguments import TexasHoldemArgument as Argument


class PublicTreeCFR:
	def __init__(self, hand_iso, solve_iter=1000, skip_iter=970):
		self.regret_epsilon = 10 ** -9
		self._cached_terminal_equities = {}
		self.hand_iso = hand_iso
		self.skip_iter = skip_iter
		self.solve_iter = solve_iter

	# run cfr iter_count times from root node with given start_range
	# @params root the root node
	# @params start_range the start range at root node, ndarray, shape(2, 1326)
	def run_cfr(self, root, start_range):
		root.ranges_absolute = start_range
		for it in range(self.solve_iter):
			self.cfr_iter_dfs(root, it)

	# run cfr depth first search
	def cfr_iter_dfs(self, node, it):
		action_dim, hole_dim = 0, 1
		cp, op = node.current_player, 1 - node.current_player
		assert (cp == Players.P0 or cp == Players.P1 or cp == Players.CHANCE)

		values = np.zeros(shape=node.ranges_absolute.shape, dtype="float")  # store cf values for this node

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
				values[0] = np.matmul(node.ranges_absolute[1], terminal_equity.call_matrix)
				values[1] = np.matmul(node.ranges_absolute[0], terminal_equity.call_matrix)

			elif node.node_type == NodeTypes.TERMINAL_FOLD:
				values[0] = np.matmul(node.ranges_absolute[1], terminal_equity.fold_matrix)
				values[1] = np.matmul(node.ranges_absolute[0], terminal_equity.fold_matrix)
				values[op, :] *= -1
			else:
				raise Exception

			values = values * node.pot
			node.cf_values = values.reshape(node.ranges_absolute.shape)

		# not terminal node
		else:

			action_count = len(node.children)
			current_strategy = None

			if node.current_player == Players.CHANCE:
				if node.strategy is not None:
					current_strategy = node.strategy
				else:
					current_strategy = np.ones(shape=(action_count, Argument.hole_count), dtype=float)
					if node.street == 0:
						current_strategy /=  (Argument.card_count - 4)
					elif node.street == 1:
						current_strategy /= (Argument.card_count - 7)
					elif node.street == 2:
						current_strategy /= (Argument.card_count - 8)
			else:
				# compute current strategy for this node

				# init regret and positive regret in first iteration
				if node.regrets is None:
					node.regrets = np.ones(shape=(action_count, Argument.hole_count), dtype=float) * self.regret_epsilon
				if node.positive_regrets is None:
					node.positive_regrets = \
						np.ones(shape=(action_count, Argument.hole_count), dtype=float) * self.regret_epsilon

				# compute positive regrets, use positive regrets to compute current strategy
				node.positive_regrets = node.regrets.copy()
				node.positive_regrets[node.positive_regrets <= 0] = self.regret_epsilon

				# compute current strategy
				regret_sum = node.positive_regrets.sum(axis=action_dim).\
					reshape((1, Argument.hole_count)).repeat(action_count, axis=action_dim)
				current_strategy = node.positive_regrets / regret_sum
			# end of computing current strategy

			# compute current cfvs [action, players, ranges]
			cf_values_allactions = np.ndarray(shape=(action_count, 2, Argument.hole_count), dtype=float)
			children_ranges_absolute = {}

			if node.current_player == Players.CHANCE:
				ranges_mul_matrix = node.ranges_absolute[0]\
					.reshape((1, Argument.hole_count)).repeat(repeats=action_count, axis=action_dim)
				children_ranges_absolute[0] = current_strategy * ranges_mul_matrix
				ranges_mul_matrix = node.ranges_absolute[1] \
					.reshape((1, Argument.hole_count)).repeat(repeats=action_count, axis=action_dim)
				children_ranges_absolute[1] = current_strategy * ranges_mul_matrix
			else:
				ranges_mul_matrix = node.ranges_absolute[cp]\
					.reshape((1, Argument.hole_count)).repeat(action_count, axis=action_dim)
				children_ranges_absolute[cp] = current_strategy * ranges_mul_matrix
				children_ranges_absolute[op] = node.ranges_absolute[op].repeat(action_count, axis=action_dim)

			for i in range(action_count):
				child_node = node.children[i]
				child_node.ranges_absolute = np.ndarray(shape=(2, Argument.hole_count), dtype=float)
				child_node.ranges_absolute[0] = children_ranges_absolute[0][i]
				child_node.ranges_absolute[1] = children_ranges_absolute[1][i]

				self.cfr_iter_dfs(child_node, it)
				cf_values_allactions[i] = child_node.cf_values

			# use cfvs from actions(children) to compute cfvs for this node
			node.cf_values = np.zeros(shape=(2, Argument.hole_count), dtype=float)

			if cp != Players.CHANCE:
				strategy_mul_matrix = current_strategy.reshape((action_count, Argument.hole_count))
				node.cf_values[cp] = (strategy_mul_matrix * cf_values_allactions[:, cp, :]).sum(axis=action_dim)
				node.cf_values[op] = (cf_values_allactions[:, op, :]).sum(axis=action_dim)
			else:
				raise Exception

			if cp != Players.CHANCE:
				# compute regrets
				current_regrets = cf_values_allactions[:, cp, :].reshape((action_count, Argument.hole_count))
				current_regrets -= node.cf_values[cp]\
					.reshape(1, Argument.hole_count).repeat(action_count, axis=action_dim)

				self.update_regrets(node, current_regrets)
				self.update_average_strategy(node, current_strategy, it)

	def update_regrets(self, node, current_regrets):
		# print(current_regrets)
		node.regrets += current_regrets
		node.regrets[node.regrets < self.regret_epsilon] = self.regret_epsilon

	def update_average_strategy(self, node, current_strategy, it):
		if it > self.skip_iter:
			action_count, action_dim = len(node.children), 0
			if node.strategy is None:
				node.strategy = np.zeros(shape=(action_count, Argument.hole_count), dtype=float)
			if node.iter_weight_sum is None:
				node.iter_weight_sum = np.zeros(shape=(1, Argument.hole_count), dtype=float)

			iter_weight_contribution = node.ranges_absolute[node.current_player]\
				.copy().reshape((1, Argument.hole_count))
			iter_weight_contribution[iter_weight_contribution <= 0] = self.regret_epsilon
			node.iter_weight_sum += iter_weight_contribution

			iter_weight = iter_weight_contribution / node.iter_weight_sum

			expanded_weight = iter_weight.repeat(repeats=action_count, axis=action_dim)
			old_strategy_scale = 1 - expanded_weight
			node.strategy *= old_strategy_scale
			node.strategy += current_strategy * expanded_weight
