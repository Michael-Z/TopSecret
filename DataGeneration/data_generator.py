# -*- coding: utf-8 -*-
import os
import torch
import pickle
import random
import numpy as np
from random import shuffle
from Equity.mask import Mask
from Settings.constants import Players
from Range.ehs import ExpectedHandStrength
from CFR.public_tree_cfr import PublicTreeCFR
from Range.range_generator import RangeGenerator
from Settings.arguments import TexasHoldemArgument as Arguments
from PokerTree.tree_builder import SimpleTreeBuilder as TreeBuilder


class DataGenerator:
    def __init__(self):
        self.save_folder = "../Data/TrainingSamples/Texas/CardValue/"
        self.round = None
        self.board_count = None
        self.save_path = None
        self.solver = None
        self.range_generator = None
        self.tree_builder = None

        self.batch_count = None
        self.deck = None

        self.batch_size = Arguments.batch_size
        self.intervals = [(100, 100), (200, 400), (400, 2000), (2000, 6000), (6000, 19950)]

        self.inputs_ph = None  		# placeholder of inputs
        self.targets_ph = None		# placeholder of targets
        self.mask_ph = None			# placeholder of mask

        self.ehs = ExpectedHandStrength(file_path="../Data/EHS/")

        self.counter = len(os.listdir(self.save_folder))

    # @param filename
    # @param rd should be 2, 3, 4
    def setup(self, rd, solve_iter=1000, skip_iter=970):
        self.round = rd
        self.board_count = [0, 3, 4, 5][rd]
        self.solver = PublicTreeCFR(solve_iter=solve_iter, skip_iter=skip_iter)
        self.range_generator = RangeGenerator(ehs=self.ehs)
        self.tree_builder = TreeBuilder(bet_sizing=None, limit_to_street=True)
        self.deck = list(range(52))

    def generate_data(self, data_count):
        self.batch_count = data_count // self.batch_size

        # initialize inputs, targets, masks' place holder
        if self.inputs_ph is None:
            self.inputs_ph = Arguments.Tensor(self.batch_size, 2 * 1326 + 1)
            self.targets_ph = Arguments.Tensor(self.batch_size, 2 * 1326)
            self.mask_ph = Arguments.Tensor(self.batch_size, 1326)

        for i in range(self.batch_count):
            # [1.0] generate random poker situations
            # board list, ranges FloatTensor, pots FloatTensor
            board, ranges, pots = self.generate_random_poker_situation()
            # ranges_tensor = Arguments.Tensor(ranges).view(Arguments.hole_count)
            # pots_tensor = Arguments.Tensor(pots).div_(Arguments.stack)
            mask = Mask.get_board_mask(board).float()  # convert ByteTensor to FloatTensor
            # mask
            self.mask_ph.copy_(mask.view(1, -1).expand_as(self.mask_ph))
            # pots of inputs
            pots.mul_(1 / Arguments.stack)
            self.inputs_ph[:, -1:].copy_(pots.view(self.batch_size, 1))
            # [2.0] solve random poker situations
            for j in range(self.batch_size):
                bet = int(pots[j])
                bets = [bet, bet]
                root = self.tree_builder.build_tree(street=self.round, initial_bets=bets,
                                                    current_player=Players.P0, board=board)
                start_range = ranges[:, j, :]  # (2, hole_count)
                self.solver.run_cfr(root, start_range=start_range)
                cf_values = root.cf_values
                # inputs
                self.inputs_ph[j, :Arguments.hole_count].copy_(ranges[0, j, :])
                self.inputs_ph[j, Arguments.hole_count:-1].copy_(ranges[1, j, :])

                # targets
                self.targets_ph[j, :Arguments.hole_count].copy_(cf_values[0])
                self.targets_ph[j, Arguments.hole_count:].copy_(cf_values[1])
                # mask already handled before loop
            # end for
            # save a batch of data
            self.save_batch_data()

    def save_batch_data(self):
        save_path = "%s%05d-%d.dat" % (self.save_folder, self.counter, self.batch_size)
        tp = (self.inputs_ph.float(), self.targets_ph.float(), self.mask_ph.float())
        with open(save_path, "wb") as f:
            pickle.dump(tp, f)
        self.counter += 1

    # @return board, ranges, pots. [one board, a batch of ranges and pots]
    # board should be a list, ranges ndarray(2 * batch_size * 1326), pots ndarray(batch_size * 1)
    def generate_random_poker_situation(self):
        """generate random poker situation, (board, ranges, pots)
        :returns board: a list of board cards
        :returns ranges: a FloatTensor (2, batch_size, hole_count) for both player ranges
        :returns pots: a FloatTensor (batch_size, ) for pots
        """
        # [1.0] generate board cards
        shuffle(self.deck)
        board = self.deck[:self.board_count]

        # [2.0] generate ranges according to board cards
        ranges = Arguments.Tensor(2, self.batch_size, Arguments.hole_count)
        self.range_generator.set_board(board=board)
        ranges[0] = torch.from_numpy(self.range_generator.generate_ranges(batch_size=self.batch_size))
        ranges[1] = torch.from_numpy(self.range_generator.generate_ranges(batch_size=self.batch_size))

        # [3.0] generate pots
        # [100, 100), [200, 400), [400, 2000), [2000, 6000), [6000, 19950)
        pots = Arguments.Tensor(self.batch_size)
        # [3.1] select uniformly from [0-4]
        for i in range(self.batch_size):
            left, right = interval = self.intervals[random.randint(0, len(self.intervals) - 1)]
            if left == right:
                pots[i] = left
            else:
                pots[i] = random.randint(left, right - 1)
        return board, ranges, pots


def main():
    dg = DataGenerator()
    dg.setup(rd=3, solve_iter=20, skip_iter=10)
    import time
    s = time.time()
    dg.generate_data(data_count=100)
    e = time.time()
    print(e - s)

if __name__ == '__main__':
    main()
