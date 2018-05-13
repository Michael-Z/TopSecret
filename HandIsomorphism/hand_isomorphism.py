# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from HandIsomorphism.utils import *

# init nth_unset equal nCr_ranks nCr_groups
nth_unset = [[0 for col in range(RANKS)] for row in range(1 << RANKS)]
equal = [[0 for col in range(SUITS)] for row in range(1 << (SUITS - 1))]
nCr_ranks = [[0 for col in range(RANKS + 1)] for row in range(RANKS + 1)]
rank_set_to_index = [0 for x in range(1 << RANKS)]
index_to_rank_set = [[0 for col in range(1 << RANKS)] for row in range(RANKS + 1)]
suit_permutations = None
nCr_groups = [[0 for col in range(SUITS + 1)] for row in range(MAX_GROUP_INDEX)]

for i in range(0, 1 << (SUITS - 1)):
    for j in range(1, SUITS):
        equal[i][j] = 1 if i & 1 << (j - 1) else 0

for i in range(0, 1 << RANKS):
    _set = ~i & (1 << RANKS) - 1
    for j in range(0, RANKS):
        nth_unset[i][j] = 0xff if _set == 0 else ctz(_set)
        _set &= _set - 1

nCr_ranks[0][0] = 1
for i in range(1, RANKS + 1):
    nCr_ranks[i][0] = nCr_ranks[i][i] = 1
    for j in range(1, i):
        nCr_ranks[i][j] = nCr_ranks[i - 1][j - 1] + nCr_ranks[i - 1][j]

nCr_groups[0][0] = 1
for i in range(1, MAX_GROUP_INDEX):
    nCr_groups[i][0] = 1
    if i < SUITS + 1:
        nCr_groups[i][i] = 1
    for j in range(1, min(i, SUITS + 1)):
        nCr_groups[i][j] = nCr_groups[i - 1][j - 1] + nCr_groups[i - 1][j]

for i in range(0, 1 << RANKS):
    _set, j = i, 1
    while _set:
        rank_set_to_index[i] += nCr_ranks[ctz(_set)][j]
        j += 1
        _set &= (_set - 1)
    index_to_rank_set[pop_count(i)][rank_set_to_index[i]] = i

num_permutations = 1
for i in range(2, SUITS + 1):
    num_permutations *= i

suit_permutations = [[0 for col in range(SUITS)] for row in range(num_permutations)]
for i in range(0, num_permutations):
    j, index, used = 0, i, 0
    while j < SUITS:
        suit = index % (SUITS - j)
        index /= SUITS - j
        index = int(index)
        shifted_suit = nth_unset[used][suit]
        suit_permutations[i][j] = shifted_suit
        used |= (1 << shifted_suit)
        j += 1


class Hand_Indexer_S:
    def __init__(self, cards_per_round):
        rounds = len(cards_per_round)
        self.cards_per_round = cards_per_round  # [MAX_ROUNDS]
        self.rounds = rounds
        # check round and card number
        self.check()

        self.permutation_to_configuration = [0] * rounds    # [rounds][]
        self.permutation_to_pi = [0] * rounds               # [rounds][]
        self.configuration_to_equal = [0] * rounds          # [rounds][]
        self.configuration = [0] * rounds                   # [rounds][][]
        self.configuration_to_suit_size = [0] * rounds      # [rounds][][]
        self.configuration_to_offset = [0] * rounds         # [rounds][][]

        self.round_start = [0] * rounds                     # [rounds]
        self.init_round_start()

        self.configurations = [0] * rounds  # [rounds]
        self.enumerate_configurations(tabulate_flag=False)   # no tabulate
        self.init_configuration_xxx()                       # init self.configuration_xxx

        # ?????????
        self.configurations = [0] * rounds                  # [rounds]
        self.enumerate_configurations(tabulate_flag=True)    # tabulate
        self.round_size = [0] * rounds                      # [rounds]
        for i in range(0, rounds):
            accum = 0
            for j in range(0, self.configurations[i]):
                next_ = accum + self.configuration_to_offset[i][j]
                self.configuration_to_offset[i][j] = accum
                accum = next_
            self.round_size[i] = accum

        self.permutations = [0] * rounds                    # [rounds]
        self.enumerate_permutations(False)

        for i in range(rounds):
            count = self.permutations[i]
            self.permutation_to_configuration[i] = [0] * count
            self.permutation_to_pi[i] = [0] * count

        self.enumerate_permutations(True)
        m = 1


    def check(self):
        if self.rounds == 0 or self.rounds > MAX_ROUNDS:
            return False
        if sum(self.cards_per_round) > CARDS:
            return False

    def init_round_start(self):
        i, j = 0, 0
        while i < self.rounds:
            self.round_start[i] = j
            j += self.cards_per_round[i]
            i += 1

    def init_configuration_xxx(self):
        for i in range(self.rounds):
            count = self.configurations[i]
            self.configuration_to_equal[i] = [0] * count
            self.configuration_to_offset[i] = [0] * count
            self.configuration[i] = [[0 for col in range(SUITS)] for row in range(count)]
            self.configuration_to_suit_size[i] = [[0 for col in range(SUITS)] for row in range(count)]

    def enumerate_configurations(self, tabulate_flag):
        used, configuration = [0] * SUITS, [0] * SUITS
        self.enumerate_configurations_r(0, self.cards_per_round[0], 0, (1 << SUITS) - 2, used, configuration,
                                        tabulate_flag)

    def enumerate_configurations_r(self, round_, remaining, suit, equal, used, configuration, tabulate_flag):
        if suit == SUITS:
            if tabulate_flag:
                self.tabulate_configurations(round_, configuration)
            else:
                self.configurations[round_] += 1

            if round_ + 1 < self.rounds:
                # advance self.cards_per_round[round_ + 1]
                self.enumerate_configurations_r(round_ + 1, self.cards_per_round[round_ + 1], 0, equal, used,
                                                configuration, tabulate_flag)
        else:
            minm = remaining if suit == SUITS - 1 else 0
            maxm = min(remaining, RANKS - used[suit])

            previous = RANKS + 1
            was_equal = equal & 1 << suit
            if was_equal:
                previous = configuration[suit - 1] >> ROUND_SHIFT * (self.rounds - round_ - 1) & ROUND_MASK
                maxm = previous if previous < maxm else maxm

            old_configuration = configuration[suit]
            old_used = used[suit]

            for i in range(minm, maxm + 1):
                new_configuration = old_configuration | i << ROUND_SHIFT * (self.rounds - round_ - 1)
                new_equal = (equal & ~(1 << suit)) | ((1 if was_equal else 0) & (i == previous)) << suit

                used[suit] = old_used + i
                configuration[suit] = new_configuration
                self.enumerate_configurations_r(round_, remaining - i, suit + 1, new_equal, used, configuration,
                                                tabulate_flag)
                configuration[suit] = old_configuration
                used[suit] = old_used
                # end for
                # end if else

    def enumerate_permutations(self, tabulate_flag):
        used, count = [0] * SUITS, [0] * SUITS
        self.enumerate_permutations_r(0, self.cards_per_round[0], 0, used, count, tabulate_flag)

    def enumerate_permutations_r(self, round_, remaining, suit, used, count, tabulate_flag):
        if suit == SUITS:
            if tabulate_flag:
                self.tabulate_permutations(round_, count)
            else:
                self.count_permutations(round_, count)

            if round_ + 1 < self.rounds:
                self.enumerate_permutations_r(round_+1, self.cards_per_round[round_+1], 0, used, count, tabulate_flag)
        else:
            minm = remaining if suit == SUITS - 1 else 0
            maxm = min(remaining, RANKS - used[suit])

            old_count, old_used = count[suit], used[suit]
            for i in range(minm, maxm + 1):
                new_count = old_count | i << ROUND_SHIFT * (self.rounds - round_ - 1)

                used[suit], count[suit] = old_used + i, new_count
                self.enumerate_permutations_r(round_, remaining - i, suit + 1, used, count, tabulate_flag)
                count[suit], used[suit] = old_count, old_used

    def count_permutations(self, round_, count):
        idx, mult = 0, 1
        for i in range(0, round_ + 1):
            j, remaining = 0, self.cards_per_round[i]
            while j < SUITS - 1:
                size = count[j] >> (self.rounds - i - 1) * ROUND_SHIFT & ROUND_MASK
                idx += mult * size
                mult *= remaining + 1
                remaining -= size
                j += 1
        # end for
        self.permutations[round_] = max(idx + 1, self.permutations[round_])

    def tabulate_configurations(self, round_, configuration):
        id_ = self.configurations[round_]
        self.configurations[round_] += 1

        out_flag = False
        while id_ > 0:
            for i in range(SUITS):
                if configuration[i] < self.configuration[round_][id_ - 1][i]:
                    break
                elif configuration[i] > self.configuration[round_][id_ - 1][i]:
                    out_flag = True
                    break
            if out_flag:
                out_flag = False
                break      # break out of outer while
            for i in range(SUITS):
                self.configuration[round_][id_-1][i] = self.configuration[round_][id_-1][i]
                self.configuration_to_suit_size[round_][id_][i] = self.configuration_to_suit_size[round_][id_-1][i]
            self.configuration_to_offset[round_][id_] = self.configuration_to_offset[round_][id_-1]
            self.configuration_to_equal[round_][id_] = self.configuration_to_equal[round_][id_-1]
            id_ -= 1

        self.configuration_to_offset[round_][id_] = 1

        assert len(configuration) == SUITS
        for i in range(SUITS):
            self.configuration[round_][id_][i] = configuration[i]

        equal_ = 0
        i = 0
        while i < SUITS:
            size = 1
            j, remaining = 0, RANKS
            while j < round_ + 1:
                ranks = configuration[i] >> ROUND_SHIFT * (self.rounds - j - 1) & ROUND_MASK
                size *= nCr_ranks[remaining][ranks]
                remaining -= ranks
                j += 1

            j = i+1
            while j < SUITS and configuration[j] == configuration[i]:
                j += 1
            for k in range(i, j):
                self.configuration_to_suit_size[round_][id_][k] = size

            self.configuration_to_offset[round_][id_] *= nCr_groups[size+j-i-1][j-i]

            for k in range(i+1, j):
                equal_ |= 1<<k

            i = j
        self.configuration_to_equal[round_][id_] = equal_ >> 1

    def tabulate_permutations(self, round_, count):
        idx, mult = 0, 1
        for i in range(0, round_+1):
            j, remaining = 0, self.cards_per_round[i]
            while j < SUITS - 1:
                size = count[j] >> (self.rounds - i - 1) * ROUND_SHIFT & ROUND_MASK
                idx += mult * size
                mult *= remaining + 1
                remaining -= size
                j += 1
        # end for

        pi = [x for x in range(SUITS)]
        for i in range(1, SUITS):
            j, pi_i = i, pi[i]
            while j > 0:
                if count[pi_i] > count[pi[j - 1]]:
                    pi[j] = pi[j-1]
                else:
                    break
                j -= 1
            pi[j] = pi_i
        # end for

        pi_idx, pi_mult, pi_used = 0, 1, 0
        for i in range(SUITS):
            this_bit = 1 << pi[i]
            smaller = pop_count((this_bit - 1) & pi_used)
            pi_idx += (pi[i] - smaller) * pi_mult
            pi_mult *= SUITS - i
            pi_used |= this_bit
        # end for

        self.permutation_to_pi[round_][idx] = pi_idx

        low, high = 0, self.configurations[round_]
        while low < high:
            mid = (low + high) // 2
            compare = 0
            for i in range(SUITS):
                that = count[pi[i]]
                other = self.configuration[round_][mid][i]
                if other > that:
                    compare = -1
                    break
                elif other < that:
                    compare = 1
                    break
            if compare == -1:
                high = mid
            elif compare == 0:
                low = high = mid
            else:
                low = mid + 1
        self.permutation_to_configuration[round_][idx] = low

    def hand_index_last(self, cards):
        indices = [0] * self.rounds
        return self.hand_index_all(cards, indices)

    def hand_index_all(self, cards, indices):
        if self.rounds > 0:
            state = Hand_Indexer_State_S()
            for i in range(self.rounds):
                indices[i] = self.hand_index_next_round(state, cards)
            return indices[self.rounds - 1]
        return 0

    def hand_index_next_round(self, state, cards):
        round_ = state.round
        state.round += 1

        ranks, shifted_ranks = [0] * SUITS, [0] * SUITS

        i, j = 0, self.round_start[round_]
        while i < self.cards_per_round[round_]:
            rank = cards[j] >> 2
            suit = cards[j] & 3
            rank_bit = 1 << rank
            ranks[suit] |= rank_bit
            shifted_ranks[suit] |= rank_bit >> pop_count((rank_bit - 1) & state.used_ranks[suit])
            i += 1
            j += 1

        for i in range(SUITS):
            used_size = pop_count(state.used_ranks[i])
            this_size = pop_count(ranks[i])
            state.suit_index[i] += state.suit_multiplier[i] * rank_set_to_index[shifted_ranks[i]]
            state.suit_multiplier[i] *= nCr_ranks[RANKS - used_size][this_size]
            state.used_ranks[i] |= ranks[i]

        i, remaining = 0, self.cards_per_round[round_]
        while i < SUITS - 1:
            this_size = pop_count(ranks[i])
            state.permutation_index += state.permutation_multiplier * this_size
            state.permutation_multiplier *= remaining + 1
            remaining -= this_size
            i += 1

        configuration = self.permutation_to_configuration[round_][state.permutation_index]
        pi_index = self.permutation_to_pi[round_][state.permutation_index]
        equal_index = self.configuration_to_equal[round_][configuration]
        offset = self.configuration_to_offset[round_][configuration]
        pi = suit_permutations[pi_index]

        suit_index, suit_multiplier = [0] * SUITS, [0] * SUITS
        for i in range(SUITS):
            suit_index[i] = state.suit_index[pi[i]]
            suit_multiplier[i] = state.suit_multiplier[pi[i]]

        index_, multiplier = offset, 1
        i = 0
        while i < SUITS:
            part, size = None, None

            if i+1 < SUITS and equal[equal_index][i+1]:
                if i+2 < SUITS and equal[equal_index][i+2]:
                    if i + 3 < SUITS and equal[equal_index][i + 3]:
                        if suit_index[i] > suit_index[i+1]:
                            suit_index[i], suit_index[i+1] = suit_index[i+1], suit_index[i]
                        if suit_index[i+2] > suit_index[i + 3]:
                            suit_index[i+2], suit_index[i + 3] = suit_index[i + 3], suit_index[i+2]
                        if suit_index[i] > suit_index[i+2]:
                            suit_index[i], suit_index[i+2] = suit_index[i+2], suit_index[i]
                        if suit_index[i+1] > suit_index[i+3]:
                            suit_index[i+1], suit_index[i+3] = suit_index[i+3], suit_index[i+1]
                        if suit_index[i+1] > suit_index[i+2]:
                            suit_index[i+1], suit_index[i+2] = suit_index[i+2], suit_index[i+1]
                        part = suit_index[i] + nCr_groups[suit_index[i + 1] + 1][2] + nCr_groups[suit_index[i + 2] + 2][
                            3] + nCr_groups[suit_index[i + 3] + 3][4]
                        size = nCr_groups[suit_multiplier[i] + 3][4]
                        i += 4
                    else:
                        if suit_index[i] > suit_index[i+1]:
                            suit_index[i], suit_index[i+1] = suit_index[i+1], suit_index[i]
                        if suit_index[i] > suit_index[i + 2]:
                            suit_index[i], suit_index[i + 2] = suit_index[i + 2], suit_index[i]
                        if suit_index[i+1] > suit_index[i + 2]:
                            suit_index[i+1], suit_index[i + 2] = suit_index[i + 2], suit_index[i+1]
                        part = suit_index[i] + nCr_groups[suit_index[i+1] + 1][2] + nCr_groups[suit_index[i+2] + 2][3]
                        size = nCr_groups[suit_multiplier[i] + 2][3]
                        i += 3
                else:
                    if suit_index[i] > suit_index[i + 1]:
                        suit_index[i], suit_index[i + 1] = suit_index[i + 1], suit_index[i]
                    part = suit_index[i] + nCr_groups[suit_index[i+1] + 1][2]
                    size = nCr_groups[suit_multiplier[i] + 1][2]
                    i += 2
            else:
                part = suit_index[i]
                size = suit_multiplier[i]
                i += 1

            index_ += multiplier * part
            multiplier *= size
        return index_
