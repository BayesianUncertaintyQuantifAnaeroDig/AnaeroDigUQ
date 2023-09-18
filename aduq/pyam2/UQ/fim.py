""" Fisher Information Matrix Uncertainty Quantification for AM2 model"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ...misc import blab
from ...uncertainty import fim, fim_pval
from .._typing import DigesterFeed, DigesterParameter, DigesterState, DigesterStates
from ..der_am2 import am2_derivative
from ..IO import err_pred_ind, param_names_to_index, parameter_dict
from ..run_am2 import run_am2


def am2_fim(
    opti_param: DigesterParameter,
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    params_eval: Optional[list[str]] = None,
    param_output: Optional[DigesterStates] = None,
    parallel: bool = True,
    silent: bool = False,
    **kwargs,
) -> dict:
    """
    Estimates parameter uncertainty for AM2 through approx Fisher information matrix.

    Args:
        opti_param, the parameter around which to compute the uncertainty
        obs, the observed output
        influent_state, initial_state, solver_method -> see run_am2 doc
        params_eval, list of parameters on which uncertainty quantification is desired.
            Default to None which amounts to every parameter
        param_output, optional, the output of AM2 for opti_param
        parallel: should computations be parallelized?
        silent: should there be no prints?

    Returns:
        A covariance matrix in standard space

    Issues:
        Uncertainty quantification for equality constrained parameters is not attempted
        The covariance matrix could be too large, so that confidence region contains impossible values.
    """
    blab(
        silent,
        """
    ====================================
    Uncertainty quantification through Fisher's information matrix
    """,
    )

    if params_eval is None:
        params_eval = list(parameter_dict.keys())

    ## Define noise level as rmse of
    # Compute prediction with param
    if param_output is None:
        param_output = run_am2(
            param=opti_param,
            influent_state=influent_state,
            initial_state=initial_state,
            **kwargs,
        )

    ## Define noise level as rmse of residuals
    residuals = np.log(param_output[:, err_pred_ind])
    sigma = np.sqrt(
        np.mean(np.log(param_output[:, err_pred_ind] / obs[:, err_pred_ind]) ** 2)
    )

    print(f"Noise level: {sigma}")

    ## Compute Jacobian of log-AM2 at param for params_eval dimension
    der_log_am2 = am2_derivative(
        param=opti_param,
        influent_state=influent_state,
        initial_state=initial_state,
        params_to_der=params_eval,
        log_am2=True,
        parallel=parallel,
        **kwargs,
    )

    ## der_am2 should be a ndarray of shape (P, T, K)
    ## P: number of parameters fitted
    ## T: number of days in output
    ## K: number of types of  (minus time)

    # Prepare_fim inputs
    err_pred_correct = [ind - 1 for ind in err_pred_ind]
    grad = der_log_am2[:, :, err_pred_correct]

    # Main function call
    fish_info, cov = fim(grad=grad, res=residuals, weights=None, sigma=sigma)

    # Prepare output: inverse of FIM, as dataframe
    cov = pd.DataFrame(cov, columns=params_eval, index=params_eval)

    return {
        "cov": cov,
        "fisher": pd.DataFrame(fish_info, columns=params_eval, index=params_eval),
        "der_log_am2": der_log_am2,
        "pred_noise": sigma,
        "opti_predict": param_output,
    }


def am2_fim_pval(
    param: DigesterParameter,
    opti_param: DigesterParameter,
    cov: Optional[pd.DataFrame] = None,
    inv_cov: Optional[pd.DataFrame] = None,
    cov_in_log: bool = False,
):
    """
    Gives a score to the Fisher information matrix UQ by stating the minimum confidence level for which the true parameter
    belongs to the confidence region.

    Args:
        true_param, the true parameter used for generating the data
        opti_param, the parameter found after optimisation
        cov, the covariance matrix found through FIM UQ
        cov_in_log, whether the covariance is in parameter log space or not

    """
    if inv_cov is None:
        uncertain_dim = cov.columns

        cov = cov.to_numpy()
        red_inv_cov = np.linalg.inv(cov)
    else:
        uncertain_dim = inv_cov.columns

        red_inv_cov = inv_cov.to_numpy()

    uncertain_ind = param_names_to_index(uncertain_dim)
    par_np = param[uncertain_ind]
    o_par_np = opti_param[uncertain_ind]

    if cov_in_log:
        par_np = np.log(par_np)
        o_par_np = np.log(o_par_np)

    return fim_pval(par_np, o_par_np, inv_cov=red_inv_cov)


def am2_fim_pred(
    opti_predict: DigesterStates,
    cov: pd.DataFrame,
    der_log_am2: np.ndarray,
    conf_lev: float = 0.99,
):
    """
    Uncertainty on predictions using Fisher's information matrix

    Should envelop the underlying signal (i.e. without taking the observation noise into
    consideration).

    der_log_am2 is assumed to be the derivative of log(AM2)

    Note:
        If part only of the parameters are considered for UQ, then der_adm1 and cov_matrix must
        have matching parameters, i.e. der_adm1 first index must coincide with the columns and row
        orders of cov_matrix. No verification is performed since der_adm1 does not store the
        parameters on which it was performed.

    Returns a dictionnary with keys:
        lower_quant: the lower quantile for the predictions
        upper_quant: the upper quantile for the predictions
        pred_var: the variance on each prediction

    Future:
        Investigate covariance structure of prediction rather than variance. Write it as two
        tensor product operation (cov(A * X) = A cov(X)A^T)) where A is the gradient at the
        calibrated parameter
    """

    # Introduce uncertainty due to parameter
    eigvl, eigvect = np.linalg.eigh(cov.to_numpy())
    pred_var = np.zeros(der_log_am2.shape[1:])

    if np.any(eigvl < 0):
        warnings.warn(
            "Part of the eigenvalues of the covariance are negative. These are set to 0"
        )
        eigvl = np.clip(eigvl, a_min=0, a_max=None)

    for i, eigenval in enumerate(eigvl):
        vect_loc = eigvect[:, i]  # eigh convention for eigenvector
        accu_loc = np.tensordot(
            der_log_am2, vect_loc, axes=(0, 0)
        )  # Should be of shape (T, K)
        pred_var += np.square(accu_loc) * eigenval
        # Point wise variance contribution of Uncertainty on parameter direction eigvect[i]

    mult_fact = norm(0, 1).ppf(0.5 + conf_lev / 2)

    pred_sigma = mult_fact * np.sqrt(pred_var)

    above_quantile = opti_predict.copy()
    above_quantile[:, 1:] = above_quantile[:, 1:] * np.exp(pred_sigma)
    below_quantile = opti_predict.copy()
    below_quantile[:, 1:] = below_quantile[:, 1:] * np.exp(-pred_sigma)

    return {
        "lower_quant": below_quantile,
        "upper_quant": above_quantile,
        "pred_var": np.append(opti_predict[:, 0:1], pred_var, axis=1),
    }


def lin_am2(
    param: DigesterParameter,
    der_log_am2: np.ndarray,
    ref_am2: DigesterStates,
    ref_param: DigesterParameter,
    params_eval: list,
):
    """
    Give a linearized version of log(AM2) close to ref_param. Only parameters in params_eval are considered
    (the derivative of AM2 with respect to other parameters is assumed to be 0).

    der_log_am2 is the derivative of log(AM2)
    """
    out = ref_am2.copy()

    delta_param = param.to_pandas() - ref_param.to_pandas()
    delta = delta_param[params_eval].to_numpy()
    mult = np.tensordot(der_log_am2, delta, axis=(2, 0))
    out[:, 1:] = out[:, 1:] * np.exp(mult)
    return out
