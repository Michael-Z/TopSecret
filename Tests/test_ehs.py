# -*- coding: utf-8 -*-
from Range.ehs import ExpectedHandStrength

ehs = ExpectedHandStrength(filename="../Range/ehs.pickle")

assert len(ehs.preflop_ehs) == 169
assert len(ehs.flop_ehs) == 1286792
assert len(ehs.turn_ehs) == 13960050
assert len(ehs.river_ehs) == 123156254

