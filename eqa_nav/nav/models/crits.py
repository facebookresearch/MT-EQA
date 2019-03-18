import torch
import torch.nn as nn

class SeqModelCriterion(nn.Module):
  def __init__(self):
    super(SeqModelCriterion, self).__init__()
  
  def forward(self, logprobs, target, mask):
    """
    Inputs:
    - logprobs  (n, L, V)
    - target    (n, L) long
    - mask      (n, L) float {0., 1.}, zeros are used to mask
    Output:
    - loss
    """
    # gather
    output = -logprobs.gather(2, target.unsqueeze(2)).squeeze(2)  # (n, L)

    # masking
    output = output * mask  # (n, L)
    output = torch.sum(output) / (torch.sum(mask) + 1e-6)
    return output

class MaskedMSELoss(nn.Module):
  def __init__(self):
    super(MaskedMSELoss, self).__init__()
  
  def forward(self, input, target, mask):
    """
    Inputs:
    - input   (n, L, d) float
    - target  (n, L, d) float
    - mask    (n, L), float {0., 1.}, zeros are used to mask
    Output:
    - loss
    """
    mask = mask.unsqueeze(2).expand_as(input)  # (n, L, d)
    loss = torch.sum( ((input - target) * mask) ** 2 )
    loss /= (torch.sum(mask) + 1e-6)
    return loss
