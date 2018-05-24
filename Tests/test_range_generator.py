# -*- coding: utf-8 -*-

import numpy as np
from Range.range_generator import RangeGenerator
from Range.ehs import ExpectedHandStrength


player_dim, hole_dim = 0, 1
board = [0, 1, 2, 3, 4]

rg = RangeGenerator(ehs=ExpectedHandStrength(file_path="../Data/EHS/"))
batch_size = 6
rg.set_board(board=board)
ranges = rg.generate_ranges(batch_size=batch_size)  # (batch_size * 1326)

assert (ranges.sum(axis=hole_dim) <= 1.0001).all()
assert (0.9999 <= ranges.sum(axis=hole_dim)).all()

uniform_ranges = rg.get_uniform_ranges()
assert (uniform_ranges - 1/1081 < 0.0001).all()

# ------------------------------------------------------------

rg.set_board(board=board[:4])
ranges = rg.generate_ranges(batch_size=batch_size)

assert (ranges.sum(axis=hole_dim) <= 1.0001).all()
assert (0.9999 <= ranges.sum(axis=hole_dim)).all()

uniform_ranges = rg.get_uniform_ranges()
assert (uniform_ranges - 1/1128 < 0.0001).all()

# ------------------------------------------------------------

rg.set_board(board=board[:3])
ranges = rg.generate_ranges(batch_size=batch_size)

assert (ranges.sum(axis=hole_dim) <= 1.0001).all()
assert (0.9999 <= ranges.sum(axis=hole_dim)).all()

uniform_ranges = rg.get_uniform_ranges()
assert (uniform_ranges - 1/1176 < 0.0001).all()

# ------------------------------------------------------------

rg.set_board(board=[])
ranges = rg.generate_ranges(batch_size=batch_size)

assert (ranges.sum(axis=hole_dim) <= 1.0001).all()
assert (0.9999 <= ranges.sum(axis=hole_dim)).all()

uniform_ranges = rg.get_uniform_ranges()
assert (uniform_ranges - 1/1326 < 0.0001).all()
