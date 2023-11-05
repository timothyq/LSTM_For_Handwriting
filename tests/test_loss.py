import pytest
import torch
from losses.custom_loss import split_and_transform_output, MDN_loss_function


@pytest.mark.loss
def test_mdn_loss_functionality():
    batch_size, seq_len, K = 10, 5, 3
    output = torch.randn(batch_size, seq_len, 6*K+1)
    y = torch.randn(batch_size, seq_len, 3)
    loss = MDN_loss_function(output, y)
    assert isinstance(loss.item(), float)


@pytest.mark.loss
def test_mdn_loss_output_shapes():
    batch_size, seq_len, K = 10, 5, 3
    output = torch.randn(batch_size, seq_len, 6*K+1)
    pi, mu1, mu2, std1, std2, corr, eot = split_and_transform_output(output)

    assert pi.shape == (batch_size, seq_len, K)
    assert mu1.shape == (batch_size, seq_len, K)
    assert mu2.shape == (batch_size, seq_len, K)
    assert std1.shape == (batch_size, seq_len, K)
    assert std2.shape == (batch_size, seq_len, K)
    assert corr.shape == (batch_size, seq_len, K)
    assert eot.shape == (batch_size, seq_len, 1)


@pytest.mark.loss
def test_mdn_loss_valid_ranges():
    batch_size, seq_len, K = 10, 5, 3
    output = torch.randn(batch_size, seq_len, 6*K+1)
    pi, mu1, mu2, std1, std2, corr, eot = split_and_transform_output(output)

    assert (pi.sum(dim=2) - 1.0).abs().max().item() < 1e-5
    assert (corr >= -1).all() and (corr <= 1).all()
    assert (eot >= 0).all() and (eot <= 1).all()
