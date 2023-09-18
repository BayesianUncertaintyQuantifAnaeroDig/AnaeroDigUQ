import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ...misc import blab
from ...uncertainty import fim, fim_pval
from ..der_adm1 import adm1_derivative
from ..IO import (
    DigesterFeed,
    DigesterInformation,
    DigesterParameter,
    DigesterState,
    DigesterStates,
    parameter_dict,
    small_predictions_numb,
)
from ..run_adm1 import run_adm1


def adm1_fim(
    opti_param: DigesterParameter,
    obs: DigesterStates,
    influent_state: DigesterFeed,
    initial_state: DigesterState,
    digester_info: DigesterInformation,
    params_eval: Optional[list[str]] = None,
    param_output: Optional[DigesterStates] = None,
    parallel: bool = True,
    silent: bool = False,
    **kwargs,
) -> dict:
    """
    Estimates parameter uncertainty for ADM1 through approximation of Fisher's information matrix
    and Cramer-Rao's lower bound on covariance of unbiased estimators.

    Note:
        Calibration procedures such as prediction error minimisation do not result in unbiased
        estimators.

    Args:
        opti_param: the parameter around which to compute the uncertainty
        obs: the observed output
        params_eval: list of parameters on which uncertainty quantification is desired
        influent_state, initial_state, digester_info, solver_method -> see run_adm1 doc
        param_output: Optional, the output of ADM1 for opti_param
        parallel: should computations be parallelized?
        silent: should there be no prints?

    Returns:
        A dictionary containing:
            covar_param: covariance matrix for parameters in params_eval, as a pd.DataFrame
            ADM1_der: the derivative of ADM1 computed during the analysis
            pred_noise: the noise level inferred
            opti_predict: the prediction of the optima

    Note:
        - The covariance matrix could be large, so that resulting confidence region contains impossible (i.e. negative) values.
        - Matrix inversion used to obtain covariance matrix could fail
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

    ## Define noise level
    # Compute prediction with param
    if param_output is None:
        param_output = run_adm1(
            param=opti_param,
            influent_state=influent_state,
            initial_state=initial_state,
            digester_info=digester_info,
            **kwargs,
        )

    ## Define noise level as rmse of residuals
    err_pred_ind = small_predictions_numb[1:]
    err_pred_correct = [ind - 1 for ind in err_pred_ind]  # Correct for removal of Time

    residuals = np.array(np.log(param_output[:, err_pred_ind] / obs[:, err_pred_ind]))

    sigma = np.sqrt(np.mean(np.array(residuals) ** 2))
    print(f"Noise level: {sigma}")

    ## Compute Jacobian of log-ADM1 at param for params_eval dimension
    der_log_adm1 = adm1_derivative(
        param=opti_param,
        params_to_der=params_eval,
        influent_state=influent_state,
        initial_state=initial_state,
        digester_info=digester_info,
        log_adm1=True,
        parallel=parallel,
        **kwargs,
    )

    ## der_adm1 should be a ndarray of shape (P, T, K)
    ## P: number of parameters in params_eval
    ## T: number of days in output
    ## K: number of types of output (not counting time)

    # Prepare fim inputs
    grad = der_log_adm1[:, :, err_pred_correct]

    # Main function call
    fish_info, cov = fim(grad=grad, res=residuals, weights=None, sigma=sigma)

    # Prepare output: inverse of FIM, as dataframe
    cov = pd.DataFrame(cov, columns=params_eval, index=params_eval)
    return {
        "cov": cov,
        "fisher": pd.DataFrame(fish_info, columns=params_eval, index=params_eval),
        "der_log_adm1": der_log_adm1,
        "pred_noise": sigma,
        "opti_predict": param_output,
    }


def adm1_fim_pval(
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
        param, the true parameter used for generating the data
        opti_param, the parameter found after optimisation
        cov, the covariance matrix found through FIM UQ. Overlooked if inv_cov_matrix is specified
            Default is None. Both cov_matrix and inv_cov_matrix can not be None
        inv_cov, the inverse of the covariance matrix found through FIM UQ.
            Default is None. Both cov_matrix and inv_cov_matrix can not be None
        cov_in_log, whether the covariance is in parameter log space or not

    """
    if inv_cov is None:
        uncertain_dim = cov.columns

        cov = cov.to_numpy()
        red_inv_cov = np.linalg.inv(cov)
    else:
        uncertain_dim = inv_cov.columns

        red_inv_cov = inv_cov.to_numpy()

    par_np = np.array(param.to_pandas()[uncertain_dim])
    o_par_np = np.array(opti_param.to_pandas()[uncertain_dim])

    if cov_in_log:
        par_np = np.log(par_np)
        o_par_np = np.log(o_par_np)

    return fim_pval(par_np, o_par_np, inv_cov=red_inv_cov)


def adm1_fim_pred(
    opti_predict: DigesterStates,
    cov: pd.DataFrame,
    der_log_adm1: np.ndarray,
    conf_lev: float = 0.99,
) -> dict:
    """
    Uncertainty on predictions using Fisher's information matrix

    Should envelop the underlying signal (i.e. without taking the observation noise into
    consideration).

    der_log_adm1 is assumed to be the derivative of log(ADM1)

    Note:
        If part only of the parameters are considered for UQ, then der_log_adm1 and cov_matrix must
        have matching parameters, i.e. der_log_adm1 first index must coincide with the columns and row
        orders of cov_matrix. No verification is performed since der_log_adm1 does not store the
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

    if np.any(eigvl < 0):
        warnings.warn(
            "Part of the eigenvalues of the covariance are negative. These are set to 0"
        )
        eigvl = np.clip(eigvl, a_min=0, a_max=None)

    pred_var = np.zeros(der_log_adm1.shape[1:])
    for i in range(der_log_adm1.shape[0]):
        vect_loc = eigvect[:, i]  # eigh convention for eigenvector
        accu_loc = np.tensordot(der_log_adm1, vect_loc, axes=(0, 0))
        pred_var += np.square(accu_loc) * eigvl[i]
        # Point wise variance contribution of covariance on parameter direction eigvect[i]

    mult_fact = norm(0, 1).ppf(0.5 + conf_lev / 2)

    pred_sigma = mult_fact * np.sqrt(pred_var)

    above_quantile = opti_predict.copy()
    above_quantile[:, 1:] = above_quantile[:, 1:] * np.exp(pred_sigma)
    below_quantile = opti_predict.copy()
    below_quantile[:, 1:] = below_quantile[:, 1:] * np.exp(-pred_sigma)
    return {
        "lower_quant": below_quantile,
        "upper_quant": above_quantile,
        "pred_var": DigesterStates(np.append(opti_predict[:, 0:1], pred_var, axis=1)),
    }


def lin_adm1(
    param: DigesterParameter,
    der_log_adm1: np.ndarray,
    ref_adm1: DigesterStates,
    ref_param: DigesterParameter,
    params_eval: list,
):
    """
    Give a linearized version of log(ADM1) close to ref_param. Only parameters in params_eval are considered
    (the derivative of ADM1 with respect to other parameters is assumed to be 0).

    der_log_adm1 is the derivative of log(ADM1)
    """
    out = ref_adm1.copy()

    delta_param = param.to_pandas() - ref_param.to_pandas()
    delta = delta_param[params_eval].to_numpy()
    mult = np.tensordot(der_log_adm1, delta, axis=(2, 0))
    out[:, 1:] = out[:, 1:] * np.exp(mult)
    return out
