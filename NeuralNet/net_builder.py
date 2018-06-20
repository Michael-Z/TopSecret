from torch.legacy import nn


class NetBuilder:

    def __init__(self):
        self.input_size = None
        self.output_size = None
        self.hidden_layer_count = None
        self.hidden_layer_size = None

    def setup_in_out(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def setup_hidden_layer(self, hidden_layer_count, hidden_layer_size):
        self.hidden_layer_count = hidden_layer_count or 3
        self.hidden_layer_size = hidden_layer_size or 1176 // 2

    # modify v1'=r1-0.5k and v2'=r2-0.5k, where k=r1v1+r2v2
    # r1v1' + r2v2' = r1(v1-0.5k) + r2(v2-0.5k)
    # = r1v1+r2v2 -0.5k(sum_r1 + sum_r2)
    # = 0
    def build_net(self):
        # [1.0] first layer
        first_layer = nn.ConcatTable()

        # [1.1] feed forward neural net, produce v1, v2
        feedforward = nn.Sequential()
        feedforward.add(nn.Linear(self.input_size, self.hidden_layer_size))
        feedforward.add(nn.PReLU())

        # add hidden layers
        for i in range(self.hidden_layer_count - 1):
            feedforward.add(nn.Linear(self.hidden_layer_size, self.hidden_layer_size))
            feedforward.add(nn.PReLU())

        feedforward.add(nn.Linear(self.hidden_layer_size, self.output_size))

        # [1.2] right part, discard pot_size, produce r1, r2
        right_part = nn.Sequential()
        right_part.add(nn.Narrow(1, 0, self.output_size))

        first_layer.add(feedforward)
        first_layer.add(right_part)

        # [2.0] outer net force counterfactual values satisfy 0-sum property
        second_layer = nn.ConcatTable()

        # accept v1,v2; ignore r1, r2
        left_part2 = nn.Sequential()
        left_part2.add(nn.SelectTable(0))

        # accept, r1,r2, v1,v2; produce -0.5k=-0.5(r1v1 + r2v2)
        right_part2 = nn.Sequential()
        right_part2.add(nn.DotProduct())
        right_part2.add(nn.Unsqueeze(1))
        right_part2.add(nn.Replicate(self.output_size, 1))
        right_part2.add(nn.Squeeze(2))
        right_part2.add(nn.MulConstant(-0.5))

        second_layer.add(left_part2)
        second_layer.add(right_part2)

        final_mlp = nn.Sequential()
        final_mlp.add(first_layer)
        final_mlp.add(second_layer)
        # accept v1,v2 and -0.5k, product v1-0.5k, v2-0.5k
        final_mlp.add(nn.CAddTable())

        return final_mlp
