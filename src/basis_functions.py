import torch

def polynomial_basis(X: torch.Tensor, degree: int = 1):
    """Apply polynomial basis transformation."""
    if degree == 1:
        return X
    
    feats = [X]
    for d in range(2, degree + 1):
        feats.append(X**d)
    return torch.cat(feats, dim=1)

def rbf_basis(X: torch.Tensor, centers: torch.Tensor, lengthscale: float = 1.0):
    """Apply RBF basis transformation: exp(-||x - c||^2 / (2 * l^2))."""
    # X: (N, D), centers: (M, D)
    # Output: (N, M)
    dist_sq = torch.cdist(X, centers)**2
    return torch.exp(-dist_sq / (2 * lengthscale**2))

def apply_basis(X: torch.Tensor, basis_type: str = 'identity', **kwargs):
    """Generic basis application wrapper."""
    if basis_type == 'identity' or basis_type is None:
        return X
    elif basis_type == 'polynomial':
        return polynomial_basis(X, degree=kwargs.get('degree', 1))
    elif basis_type == 'rbf':
        return rbf_basis(X, centers=kwargs['centers'], lengthscale=kwargs.get('lengthscale', 1.0))
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")
