import torch


class CardTool:

	@classmethod
	def float_tensor2card_list(cls, tensor):
		assert isinstance(tensor, (torch.FloatTensor, ))
		int_list = [int(x) for x in tensor.tolist()]
		return int_list

	@classmethod
	# @parameter tensor: previous rounds board cards
	# @parameter return: current round possible boards, should be a list [board1, board2, ...]
	def get_possible_future_boards(cls, tensor):
		new_boards = []
		assert isinstance(tensor, (torch.FloatTensor, ))
		old_card_list = [int(x) for x in tensor.tolist()]
		used_card_set = set(old_card_list)
		for card in range(52):
			if card in used_card_set:
				pass
			else:
				new_board_list = old_card_list[::]
				new_board_list.append(card)
				new_boards.append(new_board_list)
		return new_boards
