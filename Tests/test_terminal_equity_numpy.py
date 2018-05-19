# -*- coding: utf-8 -*-
from Equity.terminal_equity_numpy import TerminalEquity
import time


s = time.time()
te = TerminalEquity()
te.set_board(board=[1, 2, 3])
c = te.get_call_matrix()
c = te.get_fold_matrix()
e = time.time()
print(e - s)