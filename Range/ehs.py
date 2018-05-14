# -*- coding: utf-8 -*-

import pickle
from HandIsomorphism.hand_isomorphism import Hand_Indexer_S


class ExpectedHandStrength(object):

	def __init__(self, filename):
		with open(filename, 'rb') as file:
			four_round_ehs = pickle.load(file)
			self.preflop_ehs, self.flop_ehs, self.turn_ehs, self.river_ehs = four_round_ehs
		self.cards_per_round = [[2], [2, 3], [2, 4], [2, 5]]
		self.indexers = [Hand_Indexer_S(cards) for cards in self.cards_per_round]

	def get_possible_hand_ehs(self, board_cards, rd):
		ehs = [0 for i in range(1326)]
		s = set([int(x) for x in board_cards])
		for s_card in range(52 - 1):
			for b_card in range(s_card + 1, 52):
				index = b_card * (b_card - 1) // 2 + s_card
				if s_card in s or b_card in s:
					ehs[index] = -1
				else:
					cards = [s_card, b_card] + board_cards
					idx = self.indexers[rd].hand_index_last(cards)
					if rd == 0:
						ehs[index] = self.preflop_ehs[idx]
					if rd == 1:
						ehs[index] = self.flop_ehs[idx]
					if rd == 2:
						ehs[index] = self.turn_ehs[idx]
					if rd == 3:
						ehs[index] = self.river_ehs[idx]
		return ehs
