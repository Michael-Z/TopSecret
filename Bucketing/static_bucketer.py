# -*- coding: utf-8 -*-
import torch
from Range.ehs import ExpectedHandStrength


class StaticBucketer:
	def __init__(self):
		self.ehs = ExpectedHandStrength(file_path="../Data/EHS/")

	def compute_buckets(self, board):
		board_count = len(board)
		assert board_count in (0, 3, 4, 5)

		ehs_list = self.ehs.get_possible_hand_ehs(board_cards=board)
		ehs_tensor = torch.FloatTensor(ehs_list)

		# use fixed interval bucketing, convert ehs to [0,1,2...499]
		buckets = ehs_tensor.clone()
		buckets[buckets > 0].mul_(500).floor_()
