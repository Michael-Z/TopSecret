# -*- coding: utf-8 -*-
import os
import pickle
from Settings.arguments import TexasHoldemArgument as Argument
from Bucketing.static_bucketer import StaticBucketer
from Bucketing.bucket_conversion import BucketConversion


class DataConverter:
    """convert card values in raw training samples to bucket values
    raw data which needs to be converted must be stored in src_folder, '../Data/TrainingSamples/Texas/CardValue/'
    processed data would be stored in dst_folder, '../Data/TrainingSamples/Texas/BucketValue/'
    data count of each file in src folder should be 100 by default
    processed data count of each file in dst folder should also be 100 by default
    """
    def __init__(self):
        self.folder = "../Data/TrainingSamples/Texas/"
        self.src_folder = self.folder + "CardValue/"
        self.src_file_names = os.listdir(self.src_folder)
        self.src_file_count = len(self.src_file_names)
        self.dst_folder = self.folder + "BucketValue/"
        self.data_count_per_file = 100
        self.data_count = self.src_file_count * self.data_count_per_file

        self.bucketer = StaticBucketer()
        self.bucket_count = self.bucketer.bucket_count
        self.bucket_conversion = BucketConversion()
        self.counter = 0

    def convert_data(self):
        """convert raw data to real training data by converting card values to bucket values """
        # [1.0] load every raw file in Source folder
        board, inputs, targets, mask = [None] * 4
        for file_count in range(self.src_file_count):
            file_path = self.src_folder + self.src_file_names[file_count]

            with open(file_path, "rb") as f:
                board, inputs, targets, mask = pickle.load(f)
                self.bucket_conversion.set_board(board)
                # convert card range to bucket range
                p1_bucket_range = self.bucket_conversion.card_range_2_bucket_range(inputs[:, 0:Argument.hole_count])
                p2_bucket_range = self.bucket_conversion.card_range_2_bucket_range(inputs[:, Argument.hole_count:-1])
                # convert card value to bucket value
                p1_bucket_value = self.bucket_conversion.card_range_2_bucket_range(targets[:, 0:Argument.hole_count])
                p2_bucket_value = self.bucket_conversion.card_range_2_bucket_range(targets[:, Argument.hole_count:])
                # convert hole mask to bucket mask
                bucket_mask = self.bucket_conversion.get_possible_bucket_mask()  # return FloatTensor

                batch_size, hole_count_2p = targets.shape
                bucket_ranges = Argument.Tensor(batch_size, self.bucket_count * 2 + 1)
                bucket_values = Argument.Tensor(batch_size, self.bucket_count * 2)
                bucket_masks = bucket_mask.view(1, self.bucket_count).expand(batch_size, self.bucket_count)

                bucket_ranges[:, :self.bucket_count] = p1_bucket_range
                bucket_ranges[:, self.bucket_count:-1] = p2_bucket_range
                bucket_values[:, :self.bucket_count] = p1_bucket_value
                bucket_values[:, self.bucket_count:] = p2_bucket_value

                self.save_data(bucket_ranges, bucket_values, bucket_masks)

    def save_data(self, bucket_range, bucket_value, mask):
        # construct save_path using self.counter and self.batch_size
        save_path = "%s%05d-%d.dat" % (self.dst_folder, self.counter, self.data_count_per_file)
        tp = (bucket_range, bucket_value, mask)
        with open(save_path, "wb") as f:
            pickle.dump(tp, f)
        self.counter += 1


# dt = DataConverter()
# dt.convert_data()
