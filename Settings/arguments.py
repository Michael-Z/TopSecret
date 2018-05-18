import torch


class TexasHoldemArgument:

	stack = 20000
	SB = 50
	BB = 100
	Tensor = torch.FloatTensor

	hole_count = 1326
	card_count = 52

	batch_size = 100
	learning_rate = 0.01
