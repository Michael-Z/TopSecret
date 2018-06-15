# -*- coding: utf-8 -*-
import os
import pickle
from random import shuffle
from Settings.arguments import TexasHoldemArgument as Arguments


class DataGenerator:
    def __init__(self):
        self.save_folder = "../Data/TrainingSamples/Texas/CardValue/"
        self.counter = len(os.listdir(self.save_folder))  # how many data do we have for now
        self.batch_size = Arguments.batch_size

        self.board_ph = None
        self.inputs_ph = None  # placeholder of inputs
        self.targets_ph = None  # placeholder of targets
        self.mask_ph = None  # placeholder of mask

        self.round = None  # current round which we need to solve
        self.board_count = None  # how many board cards in total until current round

        self.solver = None
        self.range_generator = None
        self.tree_builder = None
        self.deck = None
        self.intervals = None
        self.ehs = None

    def setup(self, rd, solver_iters, skip_iters):
        assert 0 < rd <= 3
        self.round = rd
        self.board_count = [0, 3, 4, 5][rd]
        self.deck = list(range(52))
        self.intervals = [(100, 100), (200, 400), (400, 2000), (2000, 6000), (6000, 19950)]

        self.setup_more()

    def setup_more(self):
        pass

    def generate_data(self, data_count):
        batch_count = data_count // self.batch_size
        assert batch_count * self.batch_size == data_count

        for batch_i in range(batch_count):
            self.generate_batch_data()
            self.save_batch_data()
            self.counter += 1  # counter is used to construct file name in save_batch_data()

    def generate_batch_data(self):
        pass

    def save_batch_data(self):
        data = (self.board_ph, self.inputs_ph, self.targets_ph, self.mask_ph)
        save_path = "%s%05d-%d.dat" % (self.save_folder, self.counter, self.batch_size)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def generate_batch_random_poker_situation(self):
        pass

    def generate_board_cards(self):
        shuffle(self.deck)
        self.board_ph = self.deck[:self.board_count]

    def generate_batch_both_ranges(self, board):
        pass

    def generate_batch_pot_sizes(self):
        pass
