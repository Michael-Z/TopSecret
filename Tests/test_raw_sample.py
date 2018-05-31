# -*- coding: utf-8 -*-
import pickle
import os


src_folder = "../Data/TrainingSamples/Texas/CardValue/"
file_paths = [src_folder + file_name for file_name in os.listdir(src_folder)]


for file_path in file_paths:

    with open(file_path, "rb") as f:
        board, inputs, targets, mask = pickle.load(f)

    masks = mask.repeat(1, 2)
    masks_byte = masks.byte()
    impossible_mask_byte = 1 - masks_byte
    value_sum = targets[impossible_mask_byte].sum()
    range_sum = inputs[:, :-1][impossible_mask_byte].sum()
    assert value_sum == 0 == range_sum

    # pots
    pots = inputs[:, -1]
    assert (100 <= pots).all() and (pots <= 20000).all()
