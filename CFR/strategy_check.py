# -*- coding: utf-8 -*-
import pickle
from Equity.mask import Mask
from Settings.constants import NodeTypes, Players
from Settings.arguments import TexasHoldemArgument as Argument


class StrategyCheck:
	def __init__(self):
		self.root = None

	def load_tree(self, path):
		with open("../Data/Tree/root.dat", "rb") as f:
			self.root = pickle.load(f)
			_ = pickle.load(f)

		return

	def check(self, root=None):
		node = root or self.root
		self._check_dfs(node=node)

		return

	def _check_dfs(self, node):
		action_dim, hole_dim = 0, 1
		if node.node_type == NodeTypes.TERMINAL_CALL or node.node_type == NodeTypes.TERMINAL_FOLD:
			return

		assert node.strategy is not None

		action_count = len(node.children)
		strategy_ = node.strategy

		board_mask = Mask.get_board_mask(board=node.board)

		if node.current_player == Players.CHANCE:
			check_sum = strategy_.sum(axis=action_dim)
			assert not (strategy_ <= 0).any()
			assert not (check_sum > 1.001).any()
			assert not (check_sum < 0.999).any()

		assert (node.ranges_absolute[node.ranges_absolute < 0]).sum() == 0
		assert (node.ranges_absolute[node.ranges_absolute > 1]).sum() == 0

		# check if range consist only of cards that don't overlap with the board
		board_mask_inverse = Mask.get_board_mask_inverse(board=node.board).reshape((1, 1326))
		impossible_range_sum = node.ranges_absolute * board_mask_inverse.repeat(repeats=2, axis=0).astype(dtype=float)
		assert impossible_range_sum.sum() == 0

		for child_node in node.children:
			self._check_dfs(child_node)

		return


def main():
	sc = StrategyCheck()
	sc.load_tree("../Data/Tree/root.dat")
	sc.check()


if __name__ == "__main__":
	main()
