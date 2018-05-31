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
        self.size_function = self.dll.get_size

        self.index_function.argtypes = [c_uint32, c_uint8 * 8, c_uint8 * 7]
        self.index_function.restype = c_uint64
        self.unindex_function.argtypes = [c_uint32, c_uint8 * 8, c_uint64, c_uint8 * 7]
        self.unindex_function.restype = c_bool
        self.size_function.argtypes = [c_uint32, c_uint8 * 8]
        self.size_function.restype = c_uint64

    def setup(self, rounds, cards_per_round):
        self.rounds = rounds
        self.card_count = sum(cards_per_round)
        self.cards_per_round = (c_uint8 * 8)(*cards_per_round)

        return

    def setup_by_round(self, rounds, mode):
        if mode == "board":
            if rounds == 1:
                raise Exception
            if rounds == 2:
                self.setup(rounds=rounds, cards_per_round=[3])
            elif rounds == 3:
                self.setup(rounds=rounds, cards_per_round=[4])
            elif rounds == 4:
                self.setup(rounds=rounds, cards_per_round=[5])
            else:
                raise Exception
        elif mode == "hole":
            raise NotImplementedError

        return

    def index_hand(self, cards):
        cards_ = (c_uint8 * 7)(*cards)
        index = self.index_function(self.rounds, self.cards_per_round, cards_)

        return index

    def hand_unindex(self, index):
        cards_ = (c_uint8 * 7)(*[0] * 7)
        flag = self.unindex_function(self.rounds, self.cards_per_round, index, cards_)
        cards = [int(x) for x in cards_][0:self.card_count]

        return cards

    def get_size(self):
        size = self.size_function(self.rounds, self.cards_per_round)
        return size
