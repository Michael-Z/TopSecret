# -*- coding: utf-8 -*-

MAX_ROUNDS = 8
SUITS = 4
RANKS = 13
CARDS = 52

MAX_GROUP_INDEX = 0x100000
MAX_CARDS_PER_ROUND = 15
ROUND_SHIFT = 4
ROUND_MASK = 0xf


def ctz(number):
	s_number = bin(number)
	return len(s_number) - len(s_number.rstrip("0"))


def pop_count(number):
	return bin(number).count("1")


class Hand_Indexer_State_S:
	def __init__(self):
		self.suit_index = [0] * SUITS
		self.suit_multiplier = [0] * SUITS
		self.round = 0
		self.permutation_index = 0
		self.permutation_multiplier = 1
		self.used_ranks = [0] * SUITS
		for i in range(SUITS):
			self.suit_multiplier[i] = 1
