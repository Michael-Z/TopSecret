# -*- coding: utf-8 -*-
import torch
from Settings.arguments import TexasHoldemArgument as Arguments
from Bucketing.static_bucketer import StaticBucketer as Bucketer


class BucketConversion:
    def __init__(self):
        """initialization"""
        self.bucketer = None
        self.bucket_count = None
        self._range_matrix = None
        self._reverse_value_matrix = None

    def set_board(self, board):
        """construct _range_matrix and _reverse_value_matrix according to board
        compute bucket number of every hole, use bucket number construct conversion matrix
        :parameter board: a list a board cards
        """
        self.bucketer = Bucketer()
        self.bucket_count = self.bucketer.get_bucket_count()
        self._range_matrix = torch.FloatTensor(1326, self.bucket_count).zero_()

        buckets = self.bucketer.compute_buckets(board=board)
        buckets = Arguments.Tensor(buckets)
        class_ids = torch.arange(0, self.bucket_count)  # [0, bucket_count)
        class_ids = class_ids.cuda() if Arguments.gpu else class_ids.float()
        class_ids = class_ids.view(1, self.bucket_count).expand(1326, self.bucket_count)
        card_buckets = buckets.view(1326, 1).expand(1326, self.bucket_count)  # (1326, 500)

        self._range_matrix[torch.eq(class_ids, card_buckets)] = 1
        self._reverse_value_matrix = self._range_matrix.t().clone()  # (500, 1326)
        card_count_of_bucket = self._reverse_value_matrix.sum(1)
        self._reverse_value_matrix.div_(card_count_of_bucket.expand_as(card_count_of_bucket))

    def card_range_2_bucket_range(self, card_range):
        """compute bucket range according card range, batch operation
        :return a bucket range FloatTensor (input_batch_size, bucket_count)
        """
        return torch.mm(card_range, self._range_matrix)

    def bucket_value_2_card_value(self, bucket_value):
        """compute card value according bucket value, batch operation
        :return a bucket value FloatTensor (input_batch_size, hole_count)
        """
        return torch.mm(bucket_value, self._reverse_value_matrix)

    def get_possible_bucket_mask(self):
        """gives a vector of possible buckets on the board
        function set_board() must be called first.
        :return a bucket mask FloatTensor (1, bucket_count)
        """
        card_indicator = Arguments.Tensor(1, Arguments.hole_count).fill_(1)
        mask = torch.mm(card_indicator, self._range_matrix)
        assert mask.shape == (1, self.bucket_count)

        return mask
