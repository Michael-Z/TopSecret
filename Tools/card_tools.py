from Equity.mask import Mask
from Settings.arguments import TexasHoldemArgument as Argument


class CardTool:

    @classmethod
    def get_possible_future_boards(cls, boards):
        """compute all possible future boards
        :param boards: a list of board card
        :return: a list of all possible future boards
        """
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

    @classmethod
    def get_uniform_range(cls, board):
        """construct a range vector where all possible hands have same probability
        all impossible hands have zero probability
        :return: a uniform range which doesn't conflict with board
        """
        out = Argument.Tensor(Argument.hole_count).zero_()
        mask = Mask.get_board_mask(board)
        out[mask] = 1
        out.div_(out.sum())
        assert out.sum() == 1
        return out
