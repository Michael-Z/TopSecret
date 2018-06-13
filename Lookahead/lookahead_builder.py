# -*- coding: utf-8 -*-

from Settings import constants
from Settings.arguments import TexasHoldemArgument as Arguments
from Settings.game_setting import TexasHoldemSetting as Settings


class LookaheadBuilder:
    def __init__(self, lookahead):
        self.lookahead = lookahead

        self.lookahead.ccall_action_index = 1
        self.lookahead.fold_action_index = 0  # attention

    def construct_transition_boxes(self):
        pass

    def build_from_tree(self, tree):
        self.lookahead.tree = tree
        self.lookahead.depth = tree.depth

        self.lookahead.bets_count, self.lookahead.nonallinbets_count, self.lookahead.terminal_actions_count,\
            self.lookahead.actions_count = {}, {}, {}, {}
        self._compute_tree_structure([tree], 0)

        self.construct_data_structures()
        self.set_datastructures_from_tree_dfs(tree, 0, 0, 0, 0)

        # set additional info
        assert (self.lookahead.terminal_actions_count[0] == 1 or self.lookahead.terminal_actions_count[0] == 2)
        self.lookahead.first_call_terminal = self.lookahead.tree.children[self.lookahead.ccall_action_index].terminal
        self.lookahead.first_call_transition = self.lookahead.tree.children[
            self.lookahead.ccall_action_index].current_player == constants.Players.CHANCE
        self.lookahead.first_call_check = (not self.lookahead.first_call_terminal) and not (
            self.lookahead.first_call_transition)

        # mask out fold as a possible action when check is for free
        if self.lookahead.tree.bets[0] == self.lookahead.tree.bets[1]:
            self.lookahead.empty_action_mask[1][self.lookahead.fold_action_index].fill_(0)

        self.construct_transition_boxes()

    def _compute_tree_structure(self, current_layer, current_depth):
        """current_depth: [0, depth - 1]"""
        next_layer = []
        layer_actions_count, layer_terminal_actions_count = 0, 0

        for node in current_layer:
            layer_actions_count = max(layer_actions_count, len(node.children))
            node_terminal_actions_count = sum([1 if (child.terminal or child.current_player == constants.Players.CHANCE)
                                               else 0 for child in node.children])
            layer_terminal_actions_count = max(layer_terminal_actions_count, node_terminal_actions_count)

            # for later BFS
            next_layer.extend(node.children if not node.terminal else [])
        # end for

        assert (layer_actions_count == 0) == (len(next_layer) == 0) == (current_depth == self.lookahead.depth - 1)

        self.lookahead.bets_count[current_depth] = layer_actions_count - layer_terminal_actions_count
        self.lookahead.nonallinbets_count[
            current_depth] = layer_actions_count - layer_terminal_actions_count - 1  # - allin

        if layer_actions_count == 2:
            assert layer_actions_count == layer_terminal_actions_count
            self.lookahead.nonallinbets_count[current_depth] = 0

        self.lookahead.terminal_actions_count[current_depth] = layer_terminal_actions_count
        self.lookahead.actions_count[current_depth] = layer_actions_count

        if next_layer:
            assert layer_actions_count >= 2
            self._compute_tree_structure(next_layer, current_depth + 1)

    def construct_data_structures(self):
        self._compute_structure()

        self.lookahead.pot_size = {}
        self.lookahead.ranges_data = {}
        self.lookahead.average_strategies_data = {}
        self.lookahead.current_strategy_data = {}
        self.lookahead.cfvs_data = {}
        self.lookahead.average_cfvs_data = {}
        self.lookahead.regrets_data = {}
        self.lookahead.current_regrets_data = {}
        self.lookahead.positive_regrets_data = {}
        self.lookahead.placeholder_data = {}
        self.lookahead.regrets_sum = {}
        self.lookahead.empty_action_mask = {}  # used to mask empty actions

        # used to hold and swap inner (nonterminal) nodes when doing some transpose operations
        self.lookahead.inner_nodes = {}
        self.lookahead.inner_nodes_p1 = {}
        self.lookahead.swap_data = {}

        # create first two layers
        self.lookahead.ranges_data[0] = Arguments.Tensor(1, 1, 1, Settings.player_count, Arguments.hole_count).fill_(
            1.0 / Arguments.hole_count)
        self.lookahead.ranges_data[1] = Arguments.Tensor(self.lookahead.actions_count[0], 1, 1, Settings.player_count,
                                                         Arguments.hole_count).fill_(1.0 / Arguments.hole_count)
        self.lookahead.pot_size[0] = self.lookahead.ranges_data[0].clone().fill_(0)
        self.lookahead.pot_size[1] = self.lookahead.ranges_data[1].clone().fill_(0)
        self.lookahead.cfvs_data[0] = self.lookahead.ranges_data[0].clone().fill_(0)
        self.lookahead.cfvs_data[1] = self.lookahead.ranges_data[1].clone().fill_(0)
        self.lookahead.average_cfvs_data[0] = self.lookahead.ranges_data[0].clone().fill_(0)
        self.lookahead.average_cfvs_data[1] = self.lookahead.ranges_data[1].clone().fill_(0)
        self.lookahead.placeholder_data[0] = self.lookahead.ranges_data[0].clone().fill_(0)
        self.lookahead.placeholder_data[1] = self.lookahead.ranges_data[1].clone().fill_(0)

        # data structures for one player [actions x parent_action x grandparent_id x 1 x range]
        self.lookahead.average_strategies_data[0] = None
        self.lookahead.average_strategies_data[1] = Arguments.Tensor(self.lookahead.actions_count[0], 1, 1,
                                                                     Arguments.hole_count).fill_(0)
        self.lookahead.current_strategy_data[0] = None
        self.lookahead.current_strategy_data[1] = self.lookahead.average_strategies_data[1].clone().fill_(0)
        self.lookahead.regrets_data[0] = None
        self.lookahead.regrets_data[1] = self.lookahead.average_strategies_data[1].clone().fill_(0)
        self.lookahead.current_regrets_data[0] = None
        self.lookahead.current_regrets_data[1] = self.lookahead.average_strategies_data[1].clone().fill_(0)
        self.lookahead.positive_regrets_data[0] = None
        self.lookahead.positive_regrets_data[1] = self.lookahead.average_strategies_data[1].clone().fill_(0)
        self.lookahead.empty_action_mask[0] = None
        self.lookahead.empty_action_mask[1] = self.lookahead.average_strategies_data[1].clone().fill_(1)

        # data structures for summing over the actions [1 x parent_action x grandparent_id x range]
        self.lookahead.regrets_sum[0] = Arguments.Tensor(1, 1, 1, Arguments.hole_count).fill_(0)
        self.lookahead.regrets_sum[1] = Arguments.Tensor(1, self.lookahead.bets_count[0], 1,
                                                         Arguments.hole_count).fill_(0)

        # data structures for inner nodes (not terminal nor allin)
        # [bets_count x parent_nonallin_bets_count x gp_id x batch x players x range]
        self.lookahead.inner_nodes[0] = Arguments.Tensor(1, 1, 1, Settings.player_count, Arguments.hole_count).fill_(0)
        self.lookahead.swap_data[0] = self.lookahead.inner_nodes[0].transpose(1, 2).clone()
        self.lookahead.inner_nodes_p1[0] = Arguments.Tensor(1, 1, 1, 1, Arguments.hole_count).fill_(0)

        if self.lookahead.depth > 2:
            self.lookahead.inner_nodes[1] = Arguments.Tensor(self.lookahead.bets_count[0], 1, 1, Settings.player_count,
                                                             Arguments.hole_count).fill_(0)
            self.lookahead.swap_data[1] = self.lookahead.inner_nodes[1].transpose(1, 2).clone()
            self.lookahead.inner_nodes_p1[1] = Arguments.Tensor(self.lookahead.bets_count[0], 1, 1, 1,
                                                                Arguments.hole_count).fill_(0)

        for d in range(2, self.lookahead.depth):
            # data structures [actions x parent_action x grandparent_id x players x range]
            parents_max_branching = self.lookahead.actions_count[d - 1]
            grandparents_max_nonterminal_branching = self.lookahead.bets_count[d - 2]
            grandparents = self.lookahead.nonterminal_nonallin_nodes_count[d - 2]
            self.lookahead.ranges_data[d] = Arguments.Tensor(parents_max_branching,
                                                             grandparents_max_nonterminal_branching, grandparents,
                                                             Settings.player_count, Arguments.hole_count).fill_(0)
            self.lookahead.cfvs_data[d] = self.lookahead.ranges_data[d].clone()
            self.lookahead.placeholder_data[d] = self.lookahead.ranges_data[d].clone()
            self.lookahead.pot_size[d] = self.lookahead.ranges_data[d].clone().fill_(Arguments.stack)

        # handle rest layers
        for d in range(2, self.lookahead.depth):
            # data structures [actions x parent_action x grandparent_id x players x range]
            parents_max_branching = self.lookahead.actions_count[d - 1]
            parents_max_nonterminal_branching = self.lookahead.bets_count[d - 1]
            grandparents_max_nonterminal_branching = self.lookahead.bets_count[d - 2]
            grandparents_max_nonallin_branching = self.lookahead.nonallinbets_count[d - 2]
            grandparents = self.lookahead.nonterminal_nonallin_nodes_count[d - 2]

            self.lookahead.ranges_data[d] = Arguments.Tensor(parents_max_branching,
                                                             grandparents_max_nonterminal_branching, grandparents,
                                                             Settings.player_count, Arguments.hole_count).fill_(0)
            self.lookahead.cfvs_data[d] = self.lookahead.ranges_data[d].clone()
            self.lookahead.placeholder_data[d] = self.lookahead.ranges_data[d].clone()
            self.lookahead.pot_size[d] = self.lookahead.ranges_data[d].clone().fill_(Arguments.stack)

            # data structures [actions x parent_action x grandparent_id x 1 x range]
            self.lookahead.average_strategies_data[d] = Arguments.Tensor(parents_max_branching,
                                                                         grandparents_max_nonterminal_branching,
                                                                         grandparents, Arguments.hole_count).fill_(0)
            self.lookahead.current_strategy_data[d] = self.lookahead.average_strategies_data[d].clone()
            self.lookahead.regrets_data[d] = self.lookahead.average_strategies_data[d].clone().fill_(
                self.lookahead.regret_epsilon)
            self.lookahead.current_regrets_data[d] = self.lookahead.average_strategies_data[d].clone().fill_(0)
            self.lookahead.empty_action_mask[d] = self.lookahead.average_strategies_data[d].clone().fill_(1)
            self.lookahead.positive_regrets_data[d] = self.lookahead.regrets_data[d].clone()

            # data structures [1 x parent_action x grandparent_id x players x range]
            self.lookahead.regrets_sum[d] = Arguments.Tensor(1, grandparents_max_nonterminal_branching, grandparents,
                                                             Settings.player_count, Arguments.hole_count).fill_(0)

            # data structures for the layers except the last one
            if d < self.lookahead.depth - 1:
                self.lookahead.inner_nodes[d] = Arguments.Tensor(parents_max_nonterminal_branching,
                                                     grandparents_max_nonallin_branching,
                                                     grandparents,
                                                     Settings.player_count,
                                                     Arguments.hole_count).fill_(0)
                self.lookahead.inner_nodes_p1[d] = Arguments.Tensor(parents_max_nonterminal_branching,
                                                                    grandparents_max_nonallin_branching,
                                                                    grandparents,
                                                                    1,
                                                                    Arguments.hole_count).fill_(0)
                self.lookahead.swap_data[d] = self.lookahead.inner_nodes[d].transpose(1, 2).clone()

    def _compute_structure(self):
        assert 0 <= self.lookahead.tree.street <= 3
        self.lookahead.regret_epsilon = 1.0 / (10 ** 9)  # advance

        # acting player for each layer
        self.lookahead.acting_player = Arguments.Tensor(self.lookahead.depth).fill_(-1)
        self.lookahead.acting_player[0] = 0
        for d in range(1, self.lookahead.depth):
            self.lookahead.acting_player[d] = 1 - self.lookahead.acting_player[d - 1]

        self.lookahead.bets_count[-2], self.lookahead.bets_count[-1] = 1, 1
        self.lookahead.nonallinbets_count[-2], self.lookahead.nonallinbets_count[-1] = 1, 1
        self.lookahead.terminal_actions_count[-2], self.lookahead.terminal_actions_count[-1] = 0, 0
        self.lookahead.actions_count[-2], self.lookahead.actions_count[-1] = 1, 1

        self.lookahead.nonterminal_nodes_count = {0: 1, 1: self.lookahead.bets_count[0]}
        self.lookahead.nonterminal_nonallin_nodes_count = \
            {-1: 1, 0: 1, 1: self.lookahead.nonterminal_nodes_count[1] - 1}
        self.lookahead.all_nodes_count = {0: 1, 1: self.lookahead.actions_count[0]}
        self.lookahead.terminal_nodes_count = {0: 0, 1: 2}
        self.lookahead.allin_nodes_count = {0: 0, 1: 1}
        self.lookahead.inner_nodes_count = {0: 1, 1: 1}

        for d in range(1, self.lookahead.depth):
            # only nonterminal_nonallin_nodes at grandparent layer can reach current layer
            grandparents = self.lookahead.nonterminal_nonallin_nodes_count[d - 1]
            grandparents_max_nonterminal_branching = self.lookahead.bets_count[d - 1]
            parents_max_branching = self.lookahead.actions_count[d]

            self.lookahead.all_nodes_count[d + 1] = grandparents * grandparents_max_nonterminal_branching * \
                parents_max_branching
            self.lookahead.allin_nodes_count[d + 1] = grandparents * grandparents_max_nonterminal_branching * 1

            # only nonterminal_nonallin_nodes at grandparent layer with nonallin_action[= actions - terminal - allin]
            # can reach current layer's nonterminal_nodes
            grandparents_max_nonallin_branching = self.lookahead.nonallinbets_count[d - 1]
            parents_max_nonallin_branching = self.lookahead.nonallinbets_count[d]
            parents_max_terminal_branching = self.lookahead.terminal_actions_count[d]
            self.lookahead.nonterminal_nodes_count[d + 1] = grandparents * grandparents_max_nonallin_branching *\
                parents_max_branching
            self.lookahead.nonterminal_nonallin_nodes_count[d + 1] = grandparents * grandparents_max_nonallin_branching\
                * parents_max_nonallin_branching
            self.lookahead.terminal_nodes_count[d + 1] = grandparents * grandparents_max_nonterminal_branching *\
                parents_max_terminal_branching

            # end of function _compute_structure

    def set_datastructures_from_tree_dfs(self, node, layer, action_id, parent_id, gp_id):
        assert node.pot is not None

        self.lookahead.pot_size[layer][action_id, parent_id, gp_id, :, :] = node.pot
        node.lookahead_coordinates = Arguments.Tensor([action_id, parent_id, gp_id])

        # transition call can't not be a allin call
        if node.current_player == constants.Players.CHANCE:
            assert (parent_id < self.lookahead.nonallinbets_count[layer - 2])

        if layer < self.lookahead.depth:
            grandparents_max_nonallin_branching = self.lookahead.nonallinbets_count[layer - 2]
            parents_max_terminal_branching = self.lookahead.terminal_actions_count[layer - 1]
            # grandparents_max_terminal_branching = self.lookahead.terminal_actions_count[layer - 2]
            # parents_max_nonterminal_branching = self.lookahead.bets_count[layer - 1]

            # compute next coordinates for parent and grandparent
            next_parent_id = action_id - parents_max_terminal_branching
            next_gp_id = gp_id * grandparents_max_nonallin_branching + parent_id

            if (not node.terminal) and node.current_player != constants.Players.CHANCE:
                # parent is not allin raise
                assert parent_id < self.lookahead.nonallinbets_count[layer - 2]

                # a flag for mask empty action or not
                mask_flag = len(node.children) < self.lookahead.actions_count[layer]

                if mask_flag:
                    assert layer > 0
                    terminal_actions_count = self.lookahead.terminal_actions_count[layer]
                    assert terminal_actions_count == 2

                    existing_bets_count = len(node.children) - terminal_actions_count

                    # allin situation
                    if existing_bets_count == 0:
                        assert action_id == self.lookahead.actions_count[layer - 1] - 1

                    for child_id in range(terminal_actions_count):
                        child_node = node.children[child_id]
                        self.set_datastructures_from_tree_dfs(child_node, layer + 1, child_id, next_parent_id,
                                                              next_gp_id)

                    for b in range(existing_bets_count):
                        child_node = node.children[len(node.children) - b - 1]
                        child_id = self.lookahead.actions_count[layer] - b - 1
                        self.set_datastructures_from_tree_dfs(child_node, layer + 1, child_id, next_parent_id,
                                                            next_gp_id)
                    # mask out empty action
                    start = terminal_actions_count
                    end = self.lookahead.actions_count[layer] - existing_bets_count
                    self.lookahead.empty_action_mask[layer + 1][start:end, next_parent_id, next_gp_id, :] = 0
                else:  # full action, no need for mask'
                    for child_id in range(len(node.children)):
                        child_node = node.children[child_id]
                        self.set_datastructures_from_tree_dfs(child_node, layer + 1, child_id, next_parent_id,
                                                              next_gp_id)
