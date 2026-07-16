import torch

from ssm_lab.utils.scan import parallel_scan_log, sequential_scan


def test_scan_matches_real():
    torch.manual_seed(0)
    B, L, D = 3, 37, 5  # deliberately not a power of 2
    a = torch.rand(B, L, D) * 0.9 + 0.05
    b = torch.randn(B, L, D)
    assert torch.allclose(sequential_scan(a, b), parallel_scan_log(a, b), atol=1e-4)


def test_scan_matches_complex():
    torch.manual_seed(0)
    B, L, D = 3, 37, 5
    a = (torch.rand(B, L, D) * 0.9) * torch.exp(1j * torch.rand(B, L, D) * 6.28)
    b = torch.randn(B, L, D, dtype=torch.cfloat)
    assert torch.allclose(sequential_scan(a, b), parallel_scan_log(a, b), atol=1e-4)


def test_scan_matches_with_nonzero_h0():
    torch.manual_seed(0)
    B, L, D = 3, 20, 4
    a = torch.rand(B, L, D) * 0.9 + 0.05
    b = torch.randn(B, L, D)
    h0 = torch.randn(B, D)
    assert torch.allclose(sequential_scan(a, b, h0), parallel_scan_log(a, b, h0), atol=1e-4)


def test_scan_seq_len_one():
    """Edge case: the while-loop in parallel_scan_log should simply not execute."""
    a = torch.rand(2, 1, 3)
    b = torch.randn(2, 1, 3)
    out = parallel_scan_log(a, b)
    assert torch.allclose(out, b)
