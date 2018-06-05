# -*- coding: utf-8 -*-

import pickle
import torch
from HandIsomorphism.hand_isomorphism_encapsulation import HandIsomorphismEncapsulation as HandIsomorphism


class ExpectedHandStrength(object):

    def __init__(self, file_path):
        with open(file_path + "four_round_ehs.dat", "rb") as f:
            self.preflop_ehs, self.flop_ehs, self.turn_ehs, self.river_ehs = pickle.load(f)
        self.cards_per_round = [[2], [2, 3], [2, 4], [2, 5]]
        self.hand_indexer = HandIsomorphism()

    def get_hand_ehs(self, board):
        board_count = len(board)
        rd = [0, None, None, 1, 2, 3][board_count]

        self.hand_indexer.setup(rounds=rd + 1, cards_per_round=self.cards_per_round[rd])

        ehs = torch.FloatTensor(1326).zero_()
        used = set(board)
        for s_card in range(52 - 1):
            for b_card in range(s_card + 1, 52):
                index = b_card * (b_card - 1) // 2 + s_card
                if s_card in used or b_card in used:
                    ehs[index] = -1
                else:
                    cards = [s_card, b_card] + board
                    idx = self.hand_indexer.index_hand(cards=cards)
                    if rd == 0:
                        ehs[index] = self.preflop_ehs[idx]
                    if rd == 1:
                        ehs[index] = self.flop_ehs[idx]
                    if rd == 2:
                        ehs[index] = self.turn_ehs[idx]
                    if rd == 3:
                        ehs[index] = self.river_ehs[idx]
        return ehs
