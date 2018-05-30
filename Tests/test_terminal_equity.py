# -*- coding: utf-8 -*-
import torch
import platform
from random import shuffle, choice
from Equity.mask import Mask
from ctypes import cdll, c_int
from Equity.terminal_equity import TerminalEquity


def test_terminal_equity():
    dll = cdll.LoadLibrary("../so/hand_eval.dll") if platform.system() == "Windows" \
        else cdll.LoadLibrary("../so/hand_eval.so")
    deck = list(range(52))
    te = TerminalEquity()
    hole_mask = Mask.get_hole_mask()
    inv_mask = 1 - hole_mask

    for t in range(10):
        shuffle(deck)
        board = deck[0:5]
        te.set_board(board=board)
        call_matrix = te.call_matrix
        fold_matrix = te.fold_matrix

        _strength = (c_int * 1326)()
        dll.eval5Board((c_int * 5)(*board), 5, _strength)
        strength = torch.FloatTensor(_strength)

        assert call_matrix[inv_mask].sum() == 0
        assert (call_matrix == -call_matrix.t()).all()

        # test call matrix
        for i in range(51):
            for j in range(i + 1, 52):
                for m in range(51):
                    for n in range(m + 1, 52):
                        idx1 = (j * (j - 1)) // 2 + i
                        idx2 = (n * (n - 1)) // 2 + m
                        if m in (i, j) or n in (i, j) or i in board or j in board or m in board or n in board:
                            assert call_matrix[idx1, idx2] == 0
                        else:
                            if strength[idx1] > strength[idx2]:
                                assert call_matrix[idx1, idx2] == -1
                            elif strength[idx1] < strength[idx2]:
                                assert call_matrix[idx1, idx2] == 1
                            elif strength[idx1] == strength[idx2]:
                                assert call_matrix[idx1, idx2] == 0

        # test fold matrix
        for i in range(51):
            for j in range(i + 1, 52):
                index1 = j * (j - 1) // 2 + i
                for m in range(51):
                    for n in range(m + 1, 52):
                        index2 = n * (n - 1) // 2 + m
                    if i == m or i == n or j == m or j == n or i in board or j in board or m in board or n in board:
                        assert fold_matrix[index1, index2] == fold_matrix[index2, index1] == 0
                    else:
                        assert fold_matrix[index1, index2] == fold_matrix[index2, index1] == 1


test_terminal_equity()

