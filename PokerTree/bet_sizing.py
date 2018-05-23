from Settings.arguments import TexasHoldemArgument as Argument


class BetSizing(object):

	def __init__(self, pot_fractions=None):
		self.pot_fractions = pot_fractions or [1]

	def get_possible_bets(self, node):
		possible_bets = []
		cp = node.current_player
		p_bet = node.bets[cp]
		op_bet = node.bets[1 - cp]

		if p_bet > op_bet:
			print((p_bet, op_bet), cp)
		assert p_bet <= op_bet

		max_raise_size = Argument.stack - op_bet
		min_raise_size = op_bet - node.bets[cp]
		min_raise_size = max(min_raise_size, Argument.BB)
		min_raise_size = min(max_raise_size, min_raise_size)

		if min_raise_size == 0:
			pass
		elif min_raise_size == max_raise_size:
			bet = [op_bet, op_bet]
			bet[cp] = op_bet + min_raise_size
			possible_bets.append(bet)
		else:
			pot = 2 * op_bet
			for fraction in self.pot_fractions:
				raise_size = pot * fraction
				if min_raise_size <= raise_size < max_raise_size:
					bet = [op_bet, op_bet]
					bet[cp] = op_bet + raise_size
					possible_bets.append(bet)
			# add all-in action
			all_in_bet = [op_bet, op_bet]
			all_in_bet[cp] += max_raise_size
			possible_bets.append(all_in_bet)

		return possible_bets
