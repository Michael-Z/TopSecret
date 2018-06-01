# -*- coding: utf-8 -*-
from Bucketing.bucket_conversion import BucketConversion
import os
import pickle


src_folder = "../Data/TrainingSamples/Texas/CardValue/"
file_paths = [src_folder + file_name for file_name in os.listdir(src_folder)]

bc = BucketConversion()

for file_path in file_paths:

    with open(file_path, "rb") as f:
        board, inputs, targets, mask = pickle.load(f)

    bc.set_board(board)

    range1 = bc.card_range_2_bucket_range(inputs[:, :1326])
    assert (range1.sum(1) <= 1.0001).all()
    assert (range1.sum(1) >= 0.9999).all()

    range2 = bc.card_range_2_bucket_range(inputs[:, 1326:-1])
    assert (range2.sum(1) <= 1.0001).all()
    assert (range2.sum(1) >= 0.9999).all()

    value1 = targets[:, :1326]
    value2 = targets[:, 1326:]

    bucket_value = bc.card_range_2_bucket_range(value1)
    cv1 = bc.bucket_value_2_card_value(bucket_value)

    print((cv1 - value1).abs().sum())

    bucket_mask = bc.get_possible_bucket_mask()

    print((bucket_mask > 0).sum())

    # assert ((cv1 - value1) <= 0.1).all()
    # assert ((cv1 - value1) >= -0.1).all()

    break

