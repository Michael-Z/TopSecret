# -*- coding: utf-8 -*-
import random
import numpy as np


# range generator should generate sorted range, so it needs ehs
class RangeGenerator:

    def __init__(self, ehs):
        self.board = None
        self.range_mask = None  # a ndarray 0 for impossible(conflict)
        self.ehs = ehs
        self.ehs_list = None
        self.sorted_ehs_structure = None
        self.valid_range_width = None

    # @param board the list containing board cards
    def set_board(self, board):
        if isinstance(board, (list, )):
            self.board = board
        else:
            raise Exception

        # since board is set, we can compute EHS for each hand now
        rd = [0, None, None, 1, 2, 3][len(board)]
        self.valid_range_width = [1326, 1176, 1128, 1081][rd]

        self.ehs_list = self.ehs.get_hand_ehs(board)
        self.range_mask = np.array(self.ehs_list, dtype=int)
        self.range_mask[self.range_mask >= 0] = 1
        self.range_mask[self.range_mask == -1] = 0

        assert self.valid_range_width == self.range_mask.astype("int16").sum()
        assert rd is not None

        # we need a sorted index of ehs list, the sorted index indicates the original index of the hand
        # ehs structure = [(ehs, idx), ...], a list of tuple,
        # where first element of tuple is expected hand strength
        # second element of tuple is original index [0, 1, 2...]
        ehs_structure = list(zip(self.ehs_list, range(len(self.ehs_list))))
        # after sorting, the ehs would be like [(-1, idx), (-1, idx), ...]
        self.sorted_ehs_structure = sorted(ehs_structure, key=lambda x: x[0])

    # generate sorted ranges for one player of a batch size
    # @return ndarray(batch_size, hole_count)
    def generate_ranges(self, batch_size):
        # [1.0] generate a batch of ranges
        ranges = np.ndarray((batch_size, self.valid_range_width), dtype=float)
        probs = np.ones((batch_size, 1))
        self._generate_ranges_recursively(ranges, probs, 0, self.valid_range_width)
        # [2.0] copy ranges to each hand according to sorted index
        while_ranges = np.ndarray((batch_size, 1326), dtype=float)
        for i in range(1326 - self.valid_range_width):
            assert self.sorted_ehs_structure[i][0] == -1
            original_index = self.sorted_ehs_structure[i][1]
            while_ranges[:, original_index] = 0
        for i, j in zip(range(1326 - self.valid_range_width, 1326), range(self.valid_range_width)):
            original_index = self.sorted_ehs_structure[i][1]
            while_ranges[:, original_index] = ranges[:, j]
        return while_ranges

    # [start, end)
    # ranges will be filled with ranges which are recursively generated
    def _generate_ranges_recursively(self, ranges, probs, start, end):
        batch_size, width = ranges.shape[0], end - start
        if width == 1:
            ranges[:, start:end] = probs
        else:
            rands = np.random.random(size=(batch_size, 1))
            left_probs = probs * rands
            right_probs = probs - left_probs

            # split ranges into left parts and right parts
            se = start + end
            half_point = se // 2 if se % 2 == 0 else se // 2 + random.randint(0, 1)

            self._generate_ranges_recursively(ranges, left_probs, start, half_point)
            self._generate_ranges_recursively(ranges, right_probs, half_point, end)

    # @return self.range_mask int ndarray(1326, ), self.ehs_list list[]
    def get_mask_and_ehs_list(self):
        return self.range_mask, self.ehs_list

    def get_uniform_ranges(self):
        player_dim, hole_dim = 0, 1
        ranges = self.range_mask.reshape(1, 1326).astype(float)
        ranges /= ranges.sum()

        return ranges.repeat(repeats=2, axis=player_dim)
