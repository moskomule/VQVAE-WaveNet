import torch

from utils import digitize


def test_digitize():
    # from numpy's document
    x = torch.as_tensor([0.2, 6.4, 3.0, 1.6])
    bins = torch.as_tensor([0.0, 1.0, 2.5, 4.0, 10.0])
    out = digitize(x, bins, True)
    assert torch.equal(out, torch.as_tensor([1, 4, 3, 2], dtype=torch.int32))
