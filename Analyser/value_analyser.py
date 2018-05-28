# -*- coding: utf-8 -*-
import torch
import pickle
from Range.ehs import ExpectedHandStrength


class Analyser:
	def __init__(self):
		self.root = None
		self.ehs = ExpectedHandStrength(file_path="../Data/EHS/")

	def load_tree(self, path=None):
		with open(path or "../Data/Tree/root_tensor.dat", "rb") as f:
			self.root = pickle.load(f)

	def analyse(self, node):
		board = node.board
		values = node.cf_values
		ehs_list = self.ehs.get_possible_hand_ehs(board_cards=board, rd=3)
		for i in range(1326):
			if ehs_list[i] == -1:
				assert values[0][i] == values[1][i] == 0
		ehs_tensor = torch.FloatTensor(ehs_list)
		ehs_tensor_sorted, idx = ehs_tensor.sort(descending =True)
		values_sorted_by_ehs = values[:, idx]
		for i in range(1326):
			print(values_sorted_by_ehs[:, i])


a = Analyser()
a.load_tree()
a.analyse(a.root)
