"""Differentiable RUL regression and regularization objectives."""

import torch


def asymmetric_regression_loss(location, log_variance, target, overprediction_weight=2.0):
    """Compute the mean asymmetric location-scale regression term."""
    if location.shape != log_variance.shape or target.numel() != location.numel():
        raise ValueError("Location, log variance, and target must describe the same samples.")
    target = target.reshape_as(location)
    weight = torch.where(location > target, overprediction_weight, 1.0)
    squared_error = (target - location).square()
    return 0.5 * (log_variance + weight * squared_error * (-log_variance).exp()).mean()


def dynamics_residual_loss(time_derivative, evolution_term):
    """Compute the mean squared differential residual from supplied tensors."""
    if time_derivative.shape != evolution_term.shape:
        raise ValueError("Derivative and evolution tensors must have matching shapes.")
    return (time_derivative - evolution_term).square().mean()


def gaussian_kl_loss(mean, log_variance, free_bits=0.2):
    """Compute diagonal Gaussian KL with a per-dimension free-bits threshold."""
    if mean.ndim != 2 or mean.shape != log_variance.shape:
        raise ValueError("Gaussian parameters must have shape (batch, latent dimensions).")
    if free_bits < 0:
        raise ValueError("The free-bits threshold must be nonnegative.")
    kl_per_dimension = 0.5 * (mean.square() + log_variance.exp() - 1.0 - log_variance)
    return (kl_per_dimension - free_bits).clamp_min(0.0).sum(dim=-1).mean()


def joint_loss(regression, dynamics, kl, *, epoch, dynamics_weight=0.5,
               kl_weight=0.05, kl_warmup_epochs=10):
    """Combine regression, dynamics, and linearly annealed KL terms."""
    if epoch < 0 or dynamics_weight < 0 or kl_weight < 0:
        raise ValueError("Epoch and regularization coefficients must be nonnegative.")
    beta = kl_weight * min(1.0, epoch / max(1, kl_warmup_epochs))
    return regression + dynamics_weight * dynamics + beta * kl
