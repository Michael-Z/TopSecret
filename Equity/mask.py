# -*- coding: utf-8 -*-
import torch
from Settings.arguments import TexasHoldemArgument as Argument


class Mask:
    """compute masks
    'board mask' which used to mask out hole which are conflict with board
    'hole mask' which mask out holes which are conflict with each other

    Attributes:
        hole_mask: store hole mask once it's computed. FloatTensor (1326, 1326)
    """
    hole_mask = None

    @classmethod
    def get_hole_mask(cls):
        """return hole mask directly if it's computed, else compute it and return"""
        if cls.hole_mask is None:
            valid_hole_mask = torch.ByteTensor(Argument.hole_count, Argument.hole_count).fill_(1)

            # p1 hole:(s_card, b_card), p2 hole(s, b), p1, p2 doesn't share same card
            for s_card in range(51):
                for b_card in range(s_card + 1, 52):
                    row_index = ((b_card * (b_card - 1)) >> 1) + s_card

                    # find conflict col_index
                    for s in range(51):
                        for b in range(s + 1, 52):
                            if s == s_card or b == b_card or s == b_card or b == s_card:
                                col_index = ((b * (b - 1)) >> 1) + s
                                valid_hole_mask[row_index][col_index] = 0
            cls.hole_mask = valid_hole_mask
        return cls.hole_mask

    @classmethod
    def get_board_mask(cls, board):
        """compute board mask using given board cards
        :parameter board: a list of board cards
        :return board_mask: a ByteTensor (1326, )
        """
        s = set(board)
        board_mask = torch.ByteTensor(Argument.hole_count).fill_(1)
        for s_card in range(51):
            for b_card in range(s_card + 1, 52):
                if s_card in s or b_card in s:
                    index = ((b_card * (b_card - 1)) >> 1) + s_card
                    board_mask[index] = 0
        return board_mask
