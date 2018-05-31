# -*- coding: utf-8 -*-
import time
import torch
from Bucketing.static_bucketer import StaticBucketer
from Range.range_generator import RangeGenerator
from Range.ehs import ExpectedHandStrength as EHS

board = [0, 4, 8, 12, 22]
ehs = EHS(file_path="../Data/EHS/")
rg = RangeGenerator(ehs=ehs)
rg.set_board(board=board)
ranges = rg.get_uniform_ranges()
ranges_tensor = torch.FloatTensor(ranges)

db = StaticBucketer()
buckets = db.compute_buckets(board)

print(buckets.shape, buckets[118:122])



