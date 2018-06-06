# -*- coding: utf-8 -*-
import platform
from Equity.mask import Mask
from ctypes import cdll, c_int
from itertools import combinations
from Settings.arguments import TexasHoldemArgument as Argument

dll = cdll.LoadLibrary("../so/hand_eval.dll") if platform.system() == "Windows" \
    else cdll.LoadLibrary("../so/hand_eval.so")


class TerminalEquity(object):
    """compute terminal equity of a given board, including call_matrix and fold_matrix """
    def __init__(self):
        self.hole_mask = None
        self.inverse_hole_mask = None

        self.call_matrix = None
        self.fold_matrix = None

    def set_board(self, board):
        """compute call_matrix first, fold_matrix second by calling _set_call_matrix() and _set_fold_matrix()
        :param board: a list of board cards
        :return: None
        """
        self.compute_mask(board)
        self._set_call_matrix(board)
        self._set_fold_matrix(board)

    def compute_mask(self, board):
        """compute hole mask and it's inverse, should be called before _set_xxx_matirx()
        :param board: a list of board cards
        :return: None
        """
        self.hole_mask = Mask.get_hole_mask()
        self.inverse_hole_mask = 1 - self.hole_mask

    def _set_call_matrix(self, board):
        """call different function to compute call_matrix according to the round of a given board
        :param board: a list of board cards
        :return: None
        """
        board_count = len(board)
        self.call_matrix = Argument.Tensor(Argument.hole_count, Argument.hole_count)

        if board_count == 0:
            self.get_preflop_call_matrix(board, self.call_matrix)
        elif board_count == 3:
            self.get_flop_call_matrix(board, self.call_matrix)
        elif board_count == 4:
            self.get_turn_call_matrix(board, self.call_matrix)
        elif board_count == 5:
            self.get_last_round_call_matrix(board, self.call_matrix)
        else:
            raise Exception

    def _set_fold_matrix(self, board):
        """compute fold_matrix according to hole_mask
        :param board: a list of board cards
        :return: None
        """
        self.fold_matrix = Argument.Tensor(Argument.hole_count, Argument.hole_count)

        board_mask = Mask.get_board_mask(board)
        inverse_board_mask = 1 - board_mask
        inverse_board_mask_view1 = inverse_board_mask.view(1, -1).expand_as(self.fold_matrix)
        inverse_board_mask_view2 = inverse_board_mask.view(-1, 1).expand_as(self.fold_matrix)
        self.fold_matrix.copy_(self.hole_mask.float())
        self.fold_matrix[inverse_board_mask_view1] = 0
        self.fold_matrix[inverse_board_mask_view2] = 0

    def get_turn_call_matrix(self, board, call_matrix):
        deck = list(filter(lambda x: x not in board, range(52)))
        # enumerate
        weight_constant = 1 / len(deck)
        river_call_matrix = Argument.Tensor(Argument.hole_count, Argument.hole_count)

        for card in deck:
            new_board = board + [card]
            self.get_last_round_call_matrix(new_board, river_call_matrix)
            call_matrix.add_(river_call_matrix)

        call_matrix.mul_(weight_constant)

    def get_flop_call_matrix(self, board, call_matrix):
        deck = list(filter(lambda x: x not in board, range(52)))
        weight_constant = (len(deck) * (len(deck) - 1)) >> 1
        river_call_matrix = Argument.Tensor(Argument.hole_count, Argument.hole_count)

        for cards in combinations(deck, 2):
            new_board = board + list(cards)
            self.get_last_round_call_matrix(new_board, river_call_matrix)
            call_matrix.add_(river_call_matrix)

        call_matrix.mul_(weight_constant)

    def get_preflop_call_matrix(self, board, call_matrix):
        sample_count = 2500
        weight_constant = 1 / sample_count
        river_call_matrix = Argument.Tensor(Argument.hole_count, Argument.hole_count)
        new_boards = self.get_batch_roll_out_boards(board, sample_count)

        for new_board in new_boards:
            self.get_last_round_call_matrix(new_board, river_call_matrix)
            call_matrix.add_(river_call_matrix)

        call_matrix.mul_(weight_constant)

    @staticmethod
    def get_batch_roll_out_boards(self, board, batch_size):
        deck = list(filter(lambda x: x not in board, range(52)))
        new_boards = []
        for i in range(batch_size):
            new_boards.append(board + deck[:])
        return new_boards

    def get_last_round_call_matrix(self, board, call_matrix):
        assert len(board) == 5

        # eval every hole, get strength (1326, ) FloatTensor
        _strength = (c_int * 1326)()
        dll.eval5Board((c_int * 5)(*board), 5, _strength)
        strength = Argument.Tensor(_strength)  # -1 in strength means conflict with board

        # use strength to construct inverse board mask, rather than compute a inverse board mask, save some time
        inverse_board_mask = strength.clone().fill_(0).byte()
        inverse_board_mask[strength < 0] = 1

        assert inverse_board_mask.sum() == 1326 - 1081

        view1 = strength.view(Argument.hole_count, 1).expand_as(call_matrix)
        view2 = strength.view(1, Argument.hole_count).expand_as(call_matrix)

        call_matrix.fill_(0)
        call_matrix[view1 < view2] = 1
        call_matrix[view1 > view2] = -1
        # call_matrix[view1 == view2] = 0

        # handle blocking cards. two holes can't share same cards
        call_matrix[self.inverse_hole_mask] = 0
        # handle blocking cards. hole can't share cards with board
        inverse_board_mask_view1 = inverse_board_mask.view(1, -1).expand_as(call_matrix)
        inverse_board_mask_view2 = inverse_board_mask.view(-1, 1).expand_as(call_matrix)
        call_matrix[inverse_board_mask_view1] = 0
        call_matrix[inverse_board_mask_view2] = 0
