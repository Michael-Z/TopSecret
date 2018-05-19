

class CardTool:

	@classmethod
	# @parameter boards: previous rounds board cards, list
	# @parameter return: current round possible boards, should be a list [board1, board2, ...]
	def get_possible_future_boards(cls, boards):
		new_boards = []
		used_card_set = set(boards)
		for card in range(52):
			if card in used_card_set:
				continue
			else:
				new_board_list = boards[:]
				new_board_list.append(card)
				new_boards.append(new_board_list)
		return new_boards
