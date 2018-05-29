import torch


class TexasHoldemArgument:

    stack = 20000
    SB = 50
    BB = 100
    Tensor = torch.FloatTensor

    hole_count = 1326
    card_count = 52

    batch_size = 1000
    learning_rate = 0.01

    gpu = False
    max_number_of_regret = 10 ** 9
