# -*- coding: utf-8 -*-
from Settings.arguments import TexasHoldemAgrument


class Mask:
	# [1326 * 1326 mask out conflict hole combinations]
	hole_mask = None

	# once hole mask is computed, store it, and it won't change anymore
	# hole mask only handles player holes' conflict
	# @params return [1326 * 1326] list
	@classmethod
	def get_hole_mask(cls):
		if cls.hole_mask is None:
			hc, cc = TexasHoldemAgrument.hole_count, TexasHoldemAgrument.card_count
			valid_hole_mask = [[1 for i in range(hc)] for j in range(hc)]

			# p1 hole:(s_card, b_card), p2 hole(s, b), p1, p2 doesn't share same card
			for s_card in range(cc - 1):
				for b_card in range(s_card + 1, cc):
					row_index = b_card * (b_card - 1) // 2 + s_card

					# find conflict col_index
					for s in range(cc - 1):
						for b in range(s + 1, cc):
							if s == s_card or b == b_card or s == b_card or b == s_card:
								col_index = b * (b - 1) // 2 + s
								valid_hole_mask[row_index][col_index] = 0
			cls.hole_mask = valid_hole_mask
		return cls.hole_mask

	# return [1326] one dim vector
	# @param board should be a list
	@classmethod
	def get_board_mask(cls, board):
		s = None
		if isinstance(board, (list, )):
			s = set(board)
		else:
			raise Exception
		hc, cc = TexasHoldemAgrument.hole_count, TexasHoldemAgrument.card_count
		out = [1] * hc
		for s_card in range(cc - 1):
			for b_card in range(s_card + 1, cc):
				if s_card in s or b_card in s:
					index = b_card * (b_card - 1) // 2 + s_card
					out[index] = 0
		return out
