import torch
import torch.nn as nn

class SeqModelCriterion(nn.Module):
  def __init__(self, neg_weight=1.):
    super(SeqModelCriterion, self).__init__()
    self.neg_weight = neg_weight
  
  def forward(self, logprobs, target, mask):
    """
    Inputs:
    - logprobs (n, effective_length, V)
    - target   (n, L) {0, 1}
    - mask     (n, L) {0., 1.}
    Output:
    - loss
    """
    # chunk target / mask if logprobs is shorter
    target = target[:, :logprobs.size(1)]  
    mask = mask[:, :logprobs.size(1)]

    # make weights
    weights = (target == 0).float() * self.neg_weight + (target != 0).float()  # weights (n, L)

    # gather
    output = -logprobs.gather(2, target.unsqueeze(2)).squeeze(2)  # (n, L)

    # masking
    output =  output * weights * mask
    output = torch.sum(output) / (torch.sum(mask) + 1e-6)
    return output

