# -*- coding: utf-8 -*-
import torch
from Lookahead.result import Result
from Equity.terminal_equity import TerminalEquity
from Lookahead.lookahead_builder import LookaheadBuilder
from Settings.arguments import TexasHoldemArgument as Arguments
from Settings.game_setting import TexasHoldemSetting as Settings


class Lookahead:
    def __init__(self):
        self.builder = LookaheadBuilder(self)
        self.terminal_equity = None

    def build_lookahead(self, tree, terminal_equity=None):
        self.builder.build_from_tree(tree)
        if terminal_equity is not None:
            self.terminal_equity = terminal_equity
        else:
            self.terminal_equity = TerminalEquity()
            self.terminal_equity.set_board(tree.board)

    def resolve_first_node(self, p_range, op_range):
        self.ranges_data[0][:, :, :, 0, :].copy_(p_range)
        self.ranges_data[0][:, :, :, 1, :].copy_(op_range)
        self._compute()

    def _compute(self):
        for iters in range(Arguments.cfr_iters):
            # self._set_op_starting_range(iters)
            self._compute_current_strategies()
            self._compute_ranges()
            self._compute_update_average_strategies(iters)
            self._compute_terminal_equities()
            self._compute_cfvs()
            self._compute_regrets()
            self._compute_cumulate_average_cfvs(iters)

    def _compute_current_strategies(self):
        for d in range(1, self.depth):
            self.positive_regrets_data[d].copy_(self.regrets_data[d])
            self.positive_regrets_data[d].clamp_(self.regret_epsilon, Arguments.MAX_REGRET)

            self.positive_regrets_data[d].mul_(self.empty_action_mask[d])

            # RM
            # note that, the regret and the cfvs have switched player index
            torch.sum(self.positive_regrets_data[d], 0, keepdim=True, out=self.regrets_sum[d])
            p_current_strategy = self.current_strategy_data[d]
            player_regrets = self.positive_regrets_data[d]
            player_regrets_sum = self.regrets_sum[d]

            torch.div(player_regrets, player_regrets_sum.expand_as(player_regrets), out=p_current_strategy)

    def _compute_ranges(self):
        # strategy must be computed before ranges
        for d in range(self.depth - 1):
            cr_level_ranges, nx_level_ranges = self.ranges_data[d], self.ranges_data[d + 1]

            # parent layer
            pl_terminal_actions_count = self.terminal_actions_count[d - 1]
            pl_actions_count = self.actions_count[d - 1]
            pl_bets_count = self.bets_count[d - 1]
            assert pl_bets_count + pl_terminal_actions_count == pl_actions_count

            # grandparent layer
            gpl_nonallinbets_count = self.nonallinbets_count[d - 2]
            # gpl_terminal_actions_count = self.terminal_actions_count[d - 2]

            # copy the range of inner nodes and transpose
            p_start = pl_terminal_actions_count
            gp_end = gpl_nonallinbets_count
            transposed = cr_level_ranges[p_start:, :gp_end, :, :, :].transpose(1, 2)
            self.inner_nodes[d].copy_(transposed.view(self.inner_nodes[d].shape))

            super_view = self.inner_nodes[d].view(1, pl_bets_count, -1, Settings.player_count, Arguments.hole_count)
            super_view = super_view.expand_as(nx_level_ranges)

            nx_level_strategies = self.current_strategy_data[d + 1]
            nx_level_ranges.copy_(super_view)
            acting_player = int(self.acting_player[d])
            # ranges are swapped here, actually we should multiply (1 - acting_player), but we didn't
            nx_level_ranges[:, :, :, acting_player, :].mul_(nx_level_strategies)
        # end function _compute_ranges

    def _compute_update_average_strategies(self, iters):
        if iters > Arguments.cfr_skip_iters:
            # only compute average strategy for root node
            self.average_strategies_data[1].add_(self.current_strategy_data[1])

    def _compute_terminal_equities_terminal_equity(self):
        fold_idx = self.fold_action_index
        ccall_idx = self.ccall_action_index
        for d in range(1, self.depth):
            if Arguments.game == "leduc":
                if self.tree.street == 0:
                    raise NotImplementedError
                elif self.tree.street == 1:
                    pass
                else:
                    raise Exception
            elif Arguments.game == "texas":
                if self.tree.street == 0:
                    raise NotImplementedError
                elif self.tree.street == 1:
                    raise NotImplementedError
                elif self.tree.street == 2:
                    raise NotImplementedError
                elif self.tree.street == 3:
                    # on river, any call is terminal
                    if d > 1 or self.first_call_terminal:
                        self.terminal_equity.call_value(self.ranges_data[d][ccall_idx].view(-1, Arguments.hole_count),
                                                        self.cfvs_data[d][ccall_idx].view(-1, Arguments.hole_count))
                else:
                    raise Exception

                # folds

                self.terminal_equity.fold_value(self.ranges_data[d][fold_idx].view(-1, Arguments.hole_count),
                                                self.cfvs_data[d][fold_idx].view(-1, Arguments.hole_count))
                fold_player = int(self.acting_player[d])
                self.cfvs_data[d][0, :, :, fold_player, :].mul_(-1)

    def _compute_terminal_equities(self):
        if self.tree.street == 0:
            pass
        elif self.tree.street == 1:
            pass
        elif self.tree.street == 2:
            pass
        elif self.tree.street == 3:
            self._compute_terminal_equities_terminal_equity()
        else:
            raise Exception

        for d in range(1, self.depth):
            self.cfvs_data[d].mul_(self.pot_size[d])

    def _compute_cfvs(self):
        for d in range(self.depth - 1, 1, -1):
            gpl_terminal_actions_count = self.terminal_actions_count[d - 2]
            ggpl_nonallin_bets_count = self.nonallinbets_count[d - 3]

            self.cfvs_data[d][:, :, :, 0, :].mul_(self.empty_action_mask[d])
            self.cfvs_data[d][:, :, :, 1, :].mul_(self.empty_action_mask[d])

            self.placeholder_data[d].copy_(self.cfvs_data[d])

            # player index swapped for cfvs
            acting_player = int(self.acting_player[d])
            self.placeholder_data[d][:, :, :, acting_player, :].mul_(self.current_strategy_data[d])

            torch.sum(self.placeholder_data[d], 0, keepdim=True, out=self.regrets_sum[d])

            swap = self.swap_data[d - 1]
            swap.copy_(self.regrets_sum[d].view_as(swap))
            swap.transpose_(1, 2)
            swap = swap.view_as(self.cfvs_data[d - 1][gpl_terminal_actions_count:, :ggpl_nonallin_bets_count, :, :, :])
            self.cfvs_data[d - 1][gpl_terminal_actions_count:, :ggpl_nonallin_bets_count, :, :, :].copy_(swap)

    def _compute_cumulate_average_cfvs(self, iters):
        if iters > Arguments.cfr_skip_iters:
            self.average_cfvs_data[0].add_(self.cfvs_data[0])
            self.average_cfvs_data[1].add_(self.cfvs_data[1])

    def _compute_normalize_average_strategies(self):
        weight = 1 / (Arguments.cfr_iters - Arguments.cfr_skip_iters)
        self.average_cfvs_data[0].mul_(weight)

    def _compute_regrets(self):
        for d in range(self.depth - 1, 1, -1):
            gpl_terminal_actions_count = self.terminal_actions_count[d - 2]
            gpl_bets_count = self.bets_count[d - 2]
            ggpl_nonallin_bets_count = self.nonallinbets_count[d - 3]

            current_regrets = self.current_regrets_data[d]
            acting_player = int(self.acting_player[d])
            current_regrets.copy_(self.cfvs_data[d][:, :, :, acting_player, :])

            nx_level_cfvs = self.cfvs_data[d - 1]

            parent_inner_nodes = self.inner_nodes_p1[d - 1]
            temp = nx_level_cfvs[gpl_terminal_actions_count:, :ggpl_nonallin_bets_count,
                   :, acting_player, :].transpose(1, 2)
            temp = temp.view_as(parent_inner_nodes)
            parent_inner_nodes.copy_(temp)
            parent_inner_nodes = parent_inner_nodes.view(1, gpl_bets_count, -1, Arguments.hole_count)
            parent_inner_nodes = parent_inner_nodes.expand_as(current_regrets)

            current_regrets.sub_(parent_inner_nodes)

            torch.add(self.regrets_data[d], current_regrets, out=self.regrets_data[d])
            self.regrets_data[d].clamp_(0, Arguments.MAX_REGRET)
        # end function _compute_regrets()

    def get_results(self):
        out = Result()
        setattr(out, "strategy", None)
        setattr(out, "achieved_cfvs", None)
        setattr(out, "root_cfvs", None)
        setattr(out, "children_cfvs", None)

        # [actions x range] lookahead already computes the average strategy we just convert the dimensions
        out.strategy = self.average_strategies_data[1].view(-1, Arguments.hole_count).clone()

        # achieved opponent's CFVs at the starting node
        out.achieved_cfvs = self.average_cfvs_data[0].view(Settings.player_count, Arguments.hole_count)[0].clone()

        if hasattr(self, "reconstruction_opponent_cfvs") and self.reconstruction_opponent_cfvs:
            out.root_cfvs = None
        else:
            out.root_cfvs = self.average_cfvs_data[0].view(Settings.player_count, Arguments.hole_count)[1].clone()

            # swap cfvs indexing
            out.root_cfvs_both_players = self.average_cfvs_data[0].view(Settings.player_count,
                                                                        Arguments.hole_count).clone()
            out.root_cfvs_both_players[1].copy_(
                self.average_cfvs_data[0].view(Settings.player_count, Arguments.hole_count)[0])
            out.root_cfvs_both_players[0].copy_(
                self.average_cfvs_data[0].view(Settings.player_count, Arguments.hole_count)[1])

        # children cfvs
        out.children_cfvs = self.average_cfvs_data[1][:, :, :, 0, :].clone().view(-1, Arguments.hole_count)

        scaler = self.average_strategies_data[1].view(-1, Arguments.hole_count).clone()
        range_mul = self.ranges_data[0][:, :, :, 0, :].view(1, Arguments.hole_count).clone()
        range_mul = range_mul.expand_as(scaler)

        scaler.mul_(range_mul)
        torch.sum(scaler, 1, keepdim=True, out=scaler)
        scaler = scaler.expand_as(range_mul).clone()
        scaler.mul_(Arguments.cfr_iters - Arguments.cfr_skip_iters)

        out.children_cfvs.div_(scaler)

        return out
