"""
Uncertainty quantification using Fisher's information matrix and Cramer-Rao's lower bound.
"""
import warnings
from typing import Optional

import numpy as np
from scipy.stats import chi2


class NegativeEigenvalue(Warning):
    """
    Supposedely positive definite matrix had a negative eigenvalue.
    Usual suspect: high conditionning number resulting in numerical errors.
    """


class HighConditioningNumber(Warning):
    """
    Positive definite matrix has large conditionning number.
    One should watch out for trouble with inversion/eigen decomposition
    """


def enforce_pos(matrix: np.ndarray, epsilon=10 ** (-6)) -> np.ndarray:
    """
    Enforce positive definite condition on a matrix.

    If the smallest eigenvalue e0 of matrix is =< 0, returns matrix + (eps - e0) Id

    Args:
        matrix: should be a square matrix
        epsilon:  (default is e-8)

    Warnings issued:
        NegativeEigenvalue if eigendecomposition found negative eigenvalue
        HighConditioningNumber if conditioning number is > 1/epsilon

    """
    eigs = np.linalg.eigvalsh(matrix)

    max_eigval = np.max(np.abs(eigs))
    # matrix = matrix / max_eigval # All eigenvalues are <= 1

    if eigs[0] < 0:
        warnings.warn(
            f"""Eigendecomposition of matrix found negative eigenvalue {eigs[eigs<0]}
            Setting smallest eigenvalue to {epsilon * max_eigval}""",
            category=NegativeEigenvalue,
        )
        matrix = matrix + (max_eigval * epsilon - eigs[0]) * np.eye(matrix.shape[0])
        eigs = (
            eigs - eigs[0] + epsilon * max_eigval
        )  # Theoretical change to eigenvalues

        # This should work if the negative eigenvalues are small
        return matrix

    # Test for second potential source of error: High conditioning number
    cond = max_eigval / np.min(eigs)

    if cond > (1 / epsilon):
        warnings.warn(
            f"Conditioning number is {cond} > {1/epsilon}.",
            category=HighConditioningNumber,
        )

        to_add = max_eigval * (epsilon - 1 / cond) / (1 - epsilon)

        matrix = matrix + max_eigval * to_add * np.eye(matrix.shape[0])
        return matrix

    return matrix


def fim(
    grad: np.ndarray,
    res: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    sigma: Optional[float] = None,
) -> tuple[np.ndarray]:
    r"""
    Assuming that the statistical model is

        $Y = f(\theta) + Gaussian noise$

    computes Fisher's information matrix at $\theta$.

    Args:
        gradient, a numpy.ndarray containing the gradient $\nabla f(\theta)$ (see below for format)
        res, an optional numpy.ndarray containing the residuals (i.e. $Y - f(\theta)$)
        weights, an optional numpy.ndarray giving weights to the residuals (see below for format)
        sigma, an optional float specifying the noise level.

    One of res or sigma must be provided

    Returns:
        A tuple with the estimation of Fisher's information matrix as first element,
        and its inverse, the lower bound of the covariance for unbiased estimator

    The Gaussian noise in the model is assumed to follow $\mathcal{N}(0, sigma / \sqrt{weights})$.
    Assuming $\theta$ generated the observations, $\sqrt{weights} * res / sigma$ is distributed as
    a standard gaussian. By default, weights are assumed to be all equal to 1.0

    Input format:
        The residuals should be shaped (r1, ..., rn)
        The gradient should be shaped (p, r1, ..., rn) where p is the dimension of $\theta$
        The weights should be shaped (r1, ..., rn)

    Computes Fisher's information matrix as
        $\nabla f(\theta) diag(weights) \nabla f(\theta)^T/ (\sigma^2)$

    If sigma is not specified, it is approximated using the residuals as
        $$\sqrt{\lVert res \times \sqrt{weights} \rVert^2} / (r1 \times \dots \times rn - p)$$
    """

    # Check that res or sigma is not None
    if (res is None) and (sigma is None):
        raise ValueError("Either 'res' or 'sigma' must be specified")

    # Set default weights of 1.0 if weights are missing
    if weights is None:
        weights = np.ones(grad.shape[1:])

    n_dim = len(grad.shape)

    grad_w = grad * weights  # Shape (p, r1, ..., rn)

    pre_fim = np.tensordot(
        grad_w, grad, (tuple(range(1, n_dim)), tuple(range(1, n_dim)))
    )

    pre_fim = enforce_pos(pre_fim)
    pre_cov = np.linalg.inv(pre_fim)

    if sigma is None:
        sigma2_estim = np.sum((res**2) * weights) / (
            np.prod(res.shape) - grad.shape[0]
        )
    else:
        sigma2_estim = sigma**2

    # Computes covariance
    cov = pre_cov * sigma2_estim
    fim_mat = pre_fim / sigma2_estim

    return fim_mat, cov


def fim_pval(
    param: np.ndarray,
    mean_param: np.ndarray,
    cov: Optional[np.ndarray] = None,
    inv_cov: Optional[np.ndarray] = None,
) -> float:
    """
    Computes the p-value for the hypothesis that the parameter generating the data is param,
    from an uncertainty quantifiaction task.

    This relies on the strong assumption that the estimator has a gaussian behavior, which
    can be only safely assumed in the linear regression with gaussian noise setting.

    Args:
        param, the parameter for which to compute the p-value (test the hypothesis that the
            observations were generated from param)
        mean_param, the supposedly unbiased estimator of the parameter generating the observations
        cov, the covariance matrix found through FIM UQ. Overlooked if inv_cov is specified
            Default is None. Both cov and inv_cov can not be None.
        inv_cov, the inverse of cov (i.e. Fisher's information)
            Default is None. Both cov and inv_cov can not be None.
            Whether cov @ inv_cov = Id is not checked

    Returns: the p-value for the hypothesis that the observations were generated from param, using
        chi-square tests.
    """
    if (cov is None) and (inv_cov is None):
        raise Exception("One of cov or inv_cov must be specified")

    delta_param = mean_param - param

    if inv_cov is None:
        inv_cov = np.linalg.inv(cov)

    dist2 = np.dot(delta_param, np.matmul(inv_cov, delta_param))

    return 1 - chi2(len(delta_param)).cdf(dist2)
