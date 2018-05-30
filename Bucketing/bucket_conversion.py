# -*- coding: utf-8 -*-
import torch
from Settings.arguments import TexasHoldemArgument as Arguments
from Bucketing.static_bucketer import StaticBucketer as Bucketer


class BucketConversion:

    def __init__(self):
        self.bucketer = None
        self.bucket_count = None
        self._range_matrix = None
        self._reverse_value_matrix = None

    def set_board(self, board):
        self.bucketer = Bucketer()
        self.bucket_count = self.bucketer.get_bucket_count()
        self._range_matrix = torch.FloatTensor(1326, self.bucket_count).zero_()

        buckets = self.bucketer.compute_buckets(board=board)
        class_ids = torch.arange(0, self.bucket_count)  # [0, bucket_count)
        class_ids = class_ids.cuda() if Arguments.gpu else class_ids.float()
        class_ids = class_ids.view(1, self.bucket_count).expand(1326, self.bucket_count)
        card_buckets = buckets.view(1326, 1).expand(1326, self.bucket_count)

        self._range_matrix[torch.eq(class_ids, card_buckets)] = 1
        self._reverse_value_matrix = self._range_matrix.t().clone()

    def card_range_2_bucket_range(self, card_range):
        return torch.mm(card_range, self._range_matrix)

    # attention, special operation may needed
    def bucket_value_2_card_value(self, bucket_value, card_value):
        return torch.mm(bucket_value, self._reverse_value_matrix)
