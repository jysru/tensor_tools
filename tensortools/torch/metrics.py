import torch


def pearson(x: torch.Tensor, y: torch.Tensor, squared: bool = False, inversed: bool = False) -> torch.float:
    x, y = torch.abs(x), torch.abs(y)
    if squared:
            x, y = torch.square(x), torch.square(y)
    s = torch.sum((x - torch.mean(x)) * (y - torch.mean(y)) / torch.tensor(x.numel()).type(x.dtype))
    p = s / (torch.std(x) * torch.std(y))
    return 1 - p if inversed else p


def mae(x: torch.Tensor, y: torch.Tensor) -> torch.float:
    """Return the mean absolute error between the two arrays."""
    x, y = torch.abs(x), torch.abs(y)
    return torch.mean(torch.abs(x - y))


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.float:
    """Return the mean square error between the two arrays."""
    x, y = torch.abs(x), torch.abs(y)
    return torch.mean(torch.square(x - y))


def dot_product(x: torch.Tensor, y: torch.Tensor, normalized: bool = True) -> torch.float:
    """Return the scalar product between the two complex arrays."""
    prod = torch.sum(x * torch.conj(y))
    norm = torch.sum(torch.abs(x) * torch.abs(y)).type(prod.dtype)
    return prod / norm if normalized else prod


def quality(x: torch.Tensor, y: torch.Tensor, squared: bool = False, inversed: bool = False) -> torch.float:
    """Return the magnitude of the normalized dot product between the two complex arrays."""
    q = torch.abs(dot_product(x, y, normalized=True))
    if squared:
        q = torch.square(q)
    return 1 - q if inversed else q
