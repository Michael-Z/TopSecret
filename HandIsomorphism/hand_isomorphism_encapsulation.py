# -*- coding: utf-8 -*-
import platform
from ctypes import cdll, c_uint8, c_uint32, c_uint64, c_bool


class HandIsomorphismEncapsulation:
	def __init__(self):
		self.dll = cdll.LoadLibrary("../so/hand_index.dll") if platform.system() == "Windows" \
			else cdll.LoadLibrary("../so/hand_index.so")
		self.rounds = None
		self.cards_per_round = None
		self.card_count = None
		self.index_function = self.dll.hand_index_last_
		self.unindex_function = self.dll.hand_unindex_

		self.index_function.argtypes = [c_uint32, c_uint8 * 8, c_uint8 * 7]
		self.index_function.restype = c_uint64
		self.unindex_function.argtypes = [c_uint32, c_uint8 * 8, c_uint64, c_uint8 * 7]
		self.unindex_function.restype = c_bool

	def setup(self, rounds, cards_per_round):
		self.rounds = rounds
		self.card_count = sum(cards_per_round)
		self.cards_per_round = (c_uint8 * 8)(*cards_per_round)

	def index_hand(self, cards):
		cards_ = (c_uint8 * 7)(*cards)
		index = self.index_function(self.rounds, self.cards_per_round, cards_)

		return index

	def hand_unindex(self, index):
		cards_ = (c_uint8 * 7)(*[0] * 7)
		flag = self.unindex_function(self.rounds, self.cards_per_round, index, cards_)
		cards = [int(x) for x in cards_][0:self.card_count]

		return cards


