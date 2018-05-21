# -*- coding: utf-8 -*-
from HandIsomorphism.hand_isomorphism_encapsulation import HandIsomorphismEncapsulation as HandIsomorphism


def test_hand_isomorphism():
	hi = HandIsomorphism()
	hi.setup(rounds=1, cards_per_round=[2])

	for i in range(169):
		cards = hi.hand_unindex(index=i)
		index = hi.index_hand(cards=cards)
		assert index == i


test_hand_isomorphism()
