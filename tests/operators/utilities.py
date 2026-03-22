import torch


# fmt: off
def assert_close(
    actual: torch.Tensor, expected: torch.Tensor,
    rtol: float = 1.6e-2, atol: float = 1e-5, otol: float = 0.0,
) -> None:
    """
    Assert two tensors are close, with detailed error message on failure.

    Parameters:
    ----------
    actual: torch.Tensor
        The actual tensor.
    expected: torch.Tensor
        The expected tensor.
    rtol: float
        Relative tolerance. Default value is 1.6e-2.
    atol: float
        Absolute tolerance. Default value is 1e-5.
    otol: float
        Outlier tolerance: maximum allowed percentage of mismatched elements.
        It is between 0.0 and 1.0 with default value 0.0.

    Raises:
    ------
    AssertionError
        If tensors are not close within specified tolerances.
    """
    __tracebackhide__ = True
    close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    ofrac = (~close).float().mean().item()

    if ofrac > otol:
        lines = []
        lines.append("Tensor-likes are not close!")
        lines.append(f"  Shape: {tuple(actual.shape)}")
        lines.append(f"  Mismatch rate: {ofrac:.2%} (allowed: {otol:.2%})")
        lines.append(f"  Greatest absolute difference: {(actual - expected).abs().max():.6g}")
        raise AssertionError("\n".join(lines))
# fmt: on
