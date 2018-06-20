from torch.legacy.nn import SmoothL1Criterion
from Settings.arguments import TexasHoldemArgument
import torch


# compute a huber loss for neural net training and evaluation
# compute the loss across buckets, only the possible buckets when boards are given
class MaskedHuberLoss:

    def __init__(self):
        self.criterion = SmoothL1Criterion()
        self.mask_placeholder = None  # batch_size * feature_size(bucket number)
        self.mask_sum = None  # batch_size * 1
        self.mask_multiplier = None  # batch_size * 1

    # computes the loss over a batch of neural net outputs and targets
    # @param outputs a batch_size * 1001 tensor containing batch_size vectors of values over two player buckets, and
    # last one element is pot_size
    # @param targets a batch_size * ? tensor containing batch_size vectors of actual values over two player buckets,
    # @mask mask a vector containing batch_size mask vector generated with ?
    # @return return the sum of huber loss applied element-wise on 'outputs' and 'targets', masked so that valid buckets
    # are included
    def forward(self, outputs, targets, mask):
        batch_size = outputs.size(0)
        feature_size = outputs.size(1)

        # 1.0 zero out the outputs/targets so that the error doesn't depend on these
        outputs.mul_(mask)
        targets.mul_(mask)

        loss = self.criterion.forward(outputs, targets)

        # 2.0 if the batch size has changed, create new storage for the sum, otherwise reuse
        if self.mask_sum is None or self.mask_sum.size(0) != batch_size:
            self.mask_placeholder = TexasHoldemArgument.Tensor(mask.size()).fill_(0)
            self.mask_sum = TexasHoldemArgument.Tensor(batch_size).fill_(0)
            self.mask_multiplier = self.mask_sum.clone().fill_(0).view(-1, 1)

        # 3.0 compute mask sum for each batch
        self.mask_placeholder.copy_(mask)
        torch.sum(self.mask_placeholder, dim=1, out=self.mask_sum)
        torch.sum(mask, dim=1, out=self.mask_sum)

        # 3.1 mask multiplier - note that mask is 1 for impossible features
        self.mask_multiplier.fill_(feature_size)
        self.mask_multiplier.sub_(self.mask_sum.view(-1, 1))
        self.mask_multiplier.div_(feature_size)

        # 4.0 multiplier to get a new loss
        # loss is not really computed batch-wise correctly
        # but that does not really matter now since gradients are correct
        total_mask_sum = self.mask_sum.sum()
        element_number = batch_size * feature_size
        loss_multiplier = element_number / (element_number - total_mask_sum)
        new_loss = loss_multiplier * loss

        return new_loss

    def backward(self, outputs, targets):
        dloss_doutput = self.criterion.backward(outputs, targets)
        dloss_doutput.div_(self.mask_multiplier.expand_as(dloss_doutput))
        return dloss_doutput
