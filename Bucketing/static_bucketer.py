# -*- coding: utf-8 -*-
import torch
from Equity.mask import Mask
from Range.ehs import ExpectedHandStrength


class StaticBucketer:
    """compute buckets of the board using expected hand strength"""
    def __init__(self):
        """initialize expect hand strength and bucket count"""
        self.ehs = ExpectedHandStrength(file_path="../Data/EHS/")
        self.bucket_count = 500

    def compute_buckets(self, board):
        """compute buckets using expected hand strength
        :parameter board: a list of board card
        :return a buckets FloatTensor (bucket_count, )
        """
        board_count = len(board)
        rd = [0, None, None, 1, 2, 3][board_count]
        assert board_count in (0, 3, 4, 5)

        ehs_list = self.ehs.get_possible_hand_ehs(board, rd)
        ehs_tensor = torch.FloatTensor(ehs_list)
        board_mask = Mask.get_board_mask(board)
        # use fixed interval bucketing, convert ehs to [0,1,2...499]
        buckets = ehs_tensor.clone()
        buckets[board_mask] *= 500
        buckets.floor_()

        return buckets

    def get_bucket_count(self):
        """get total bucket count
        :return the number of buckets
        """
        return self.bucket_count
