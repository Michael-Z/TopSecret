# -*- coding: utf-8 -*-
from Lookahead.look_ahead import Lookahead
from PokerTree.tree_builder import SimpleTreeBuilder


class Resolving:
    def __init__(self):
        # use default bet sizing [FCPA]
        self.tree_builder = SimpleTreeBuilder(bet_sizing=None, limit_to_street=True)
        self.lookahead_tree = None  # root node of the lookahead tree
        self.lookahead = None  # Lookahead instance
        self.resolve_results = None  # store the result of resolving

    def _create_lookahead_tree(self, street, initial_bets, current_player, board):
        """build a tree from given game state(street, initial bets, current player and board)
        depth-limited public tree
        :parameter street: current betting round
        :parameter initial_bets: a list of spent chips of both players
        :parameter current_player: the current acting player, 0 or 1
        :parameter board: a list of board cards
        :return None
        """
        self.lookahead_tree = self.tree_builder.build_tree(street, initial_bets, current_player, board)
        return

    def resolve_first_node(self, node, player_range, opponent_range):
        """re-solves a depth-limited lookahead using input ranges of both players
        use the input range for the opponent instead of a gadget range, so only appropriate for re-solving
        the root node of the game tree(where ranges are fixed)
        :param node: the public tree node at which to re-solve
        :param player_range: a range vector for the re-solving player, FloatTensor
        :param opponent_range: a range vector for the opponent, FloatTensor
        :return: the re-solve result
        """
        self._create_lookahead_tree(node.street, node.bets, node.current_player, node.board)
        self.lookahead = Lookahead()
        self.lookahead.build_lookahead(self.lookahead_tree)  # build lookahead data structure

        self.lookahead.resolve_first_node(player_range, opponent_range)
        self.resolve_results = self.lookahead.get_results()
        return self.resolve_result

    def resolve(self, node, player_range, opponent_cfvs):
        """re-solves a depth-limited lookahead using an input range for the player and the CFR-D to generate
        ranges for the opponent
        :param node: the public tree node at which to re-solve
        :param player_range: a range vector for the re-solving player, FloatTensor
        :param opponent_cfvs: a opponent cfv vector achieved by the opponent before re-solving
        :return: the re-solve result
        """
        self._create_lookahead_tree(node.street, node.bets, node.current_player, node.board)
        self.lookahead = Lookahead()
        self.lookahead.build_lookahead(self.lookahead_tree)

        self.lookahead.resolve(player_range, opponent_cfvs)
        self.resolve_results = self.lookahead.get_results()
        return self.resolve_result

    def _action_2_action_id(self, action):
        """Gives the index of the given action at the node being re-solved.
        The node must first be re-solved with @{resolve} or @{resolve_first_node}.
        :param action: a legal action at the node
        :return: the index of the action
        """
        actions = self.get_possible_actions()
        for i in range(len(actions)):
            if action == actions[i]:
                return i
        raise Exception

    def get_possible_actions(self):
        """Gives a list of possible actions at the node being re-solved.
        The node must first be re-solved with @{resolve} or @{resolve_first_node}.
        :return: a list of legal actions
        """
        return self.lookahead_tree.actions

    def get_root_cfv(self):
        """Gives the average counterfactual values that the re-solve player received at the node during re-solving.
        The node must first be re-solved with @{resolve_first_node}.
        :return: a vector of cfvs
        """
        return self.resolve_results.root_cfvs

    def get_root_cfv_both_players(self):
        """ Gives the average counterfactual values that each player received at the node during re-solving.
        Useful for data generation for neural net training
        The node must first be re-solved with @{resolve_first_node}.
        :return: a 2xK tensor of cfvs, where K is the range size
        """
        return self.resolve_results.root_cfvs_both_players

    def get_action_cfv(self, action):
        """Gives the average counterfactual values that the opponent received during re-solving after the re-solve
        player took a given action.
        Used during continual re-solving to track opponent cfvs. The node must first be re-solved with @{resolve} or
        @{resolve_first_node}.
        :param action:  the action taken by the re-solve player at the node being re-solved
        :return: a vector of cfvs
        """
        action_id = self._action_2_action_id(action)
        return self.resolve_results.children_cfvs[action_id]

    def get_chance_action_cfv(self, action, board):
        """Gives the average counterfactual values that the opponent received during re-solving after a chance event
        (the betting round changes and more cards are dealt).
        Used during continual re-solving to track opponent cfvs. The node must first be re-solved with @{resolve} or
        @{resolve_first_node}.
        :param action: the action taken by the re-solve player at the node being re-solved
        :param board: a vector of board cards which were updated by the chance event
        :return: a vector of cfvs
        """
        action_id = self._action_2_action_id(action)
        return self.lookahead.get_chance_action_cfv(action_id, board)

    def get_action_strategy(self, action):
        """Gives the probability that the re-solved strategy takes a given action.
        The node must first be re-solved with @{resolve} or @{resolve_first_node}.
        :param action:  a legal action at the re-solve node
        :return: a vector giving the probability of taking the action with each private hand
        """
        action_id = self._action_2_action_id(action)
        return self.resolve_results.strategy[action_id]
