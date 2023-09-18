from typing import List

import numpy as np
import pandas as pd
from scipy.stats import gamma, qmc

from ..IO import FreeDigesterParameter, parameter_dict

param_list = list(parameter_dict.keys())


def gamma_transform(mode: float, sigma: float) -> tuple:
    alpha = (mode / sigma) ** 2
    k = 1 + alpha * (1 + np.sqrt(1 + 4 / alpha)) / 2
    theta = mode / (k - 1)
    return (k, theta)


def gamma_ppf(q: float, mode: float, sigma: float) -> float:
    k, theta = gamma_transform(mode, sigma)
    return theta * gamma.ppf(q, a=k)


def min_max_distrib(prior_line: pd.Series) -> List[float]:
    """
    Transform a distribution description into an interval range.

    For Uniform distribution, interval is not changed
    For gamma/normal distributions, interval is mode +- standard_deviation.
    If standard_deviation > mean, then the minimum is mean * 10**-2
    (0 would create numerical issues)
    """
    draw_type = prior_line["distrib"]
    if draw_type == "gamma":
        mode, sigma = prior_line[["mode", "sd_in_lin"]]
        v_min = max(mode * 10**-2, mode - sigma)
        v_max = mode + sigma
    elif draw_type == "normal":
        mu, sigma = prior_line[["mean_in_lin", "sd_in_lin"]]
        v_min = max(mu * 10**-2, mu - sigma)
        v_max = mu + sigma
    elif draw_type == "unif":
        low, high = prior_line[["min_in_lin", "max_in_lin"]]
        v_min = low
        v_max = high
    elif draw_type == "log-normal":
        mu, sigma = prior_line[["mean_in_lin", "sd_in_lin"]]
        v_min = 10 ** (mu - sigma)
        v_max = 10 ** (mu + sigma)
    elif draw_type == "log-unif":
        low, high = prior_line[["min_in_lin", "max_in_lin"]]
        v_min = 10**low
        v_max = 10**high
    return [v_min, v_max]


def transform_prior(prior: pd.DataFrame) -> pd.DataFrame:
    """Transform description of tensorized priors into range values used for morris sensitivity analysis."""
    return pd.DataFrame(
        [min_max_distrib(prior.iloc[i]) for i in range(prior.shape[0])],
        index=prior.index,
        columns=["min", "max"],
    )


def multi_dim_samp(dim: int, k: int) -> List[List[int]]:
    combi = qmc.Sobol(dim).random(k) * k
    return [[int(val) for val in vals] for vals in combi.T]


def get_values(k: int, param_range: pd.DataFrame) -> pd.DataFrame:
    """From a parameter range (i.e., per parameter, a min and max value),
    outputs a dataframe with k columns giving k different levels for each parameter."""
    # print(param_range)

    return pd.DataFrame(
        np.linspace(param_range["min"], param_range["max"], k).T,
        index=param_range.index,
    )


def generate_morris_lines(
    r: int, n_lev: int, param_range: pd.DataFrame
) -> List[FreeDigesterParameter]:
    """
    Generate r morris lines.
    A morris line is a sequence of parameters such that only one dimension of parameter is changed,
    in a similar way, at each iteration.
    The increments per dimension are estimated from the prior object, which is transformed into a min/max
    information per distribution, then this interval is split into n_lev regular values.

    Outputs a List of List of one dimensional array (a parameter which can be directly fed to run_adm1)
    """
    helper_values = get_values(n_lev, param_range)

    n_par = len(param_list)

    def generate_one_line() -> List[np.ndarray]:
        initial_val_numb = np.random.choice(n_lev, n_par, replace=True)
        initial = np.zeros(n_par)

        # Draw an initial
        for i, (pg_i, vn_i) in enumerate(zip(param_list, initial_val_numb)):
            initial[i] = helper_values.loc[pg_i, vn_i]

        ## End of initialisation
        accu = np.zeros((n_par + 1, n_par))
        accu[0] = initial

        for i, (pg_i, vn_i) in enumerate(zip(param_list, initial_val_numb)):
            if initial_val_numb[i] == 0:
                initial[i] = helper_values.loc[pg_i, 1]
            elif initial_val_numb[i] == (n_lev - 1):
                initial[i] = helper_values.loc[pg_i, n_lev - 2]
            else:
                u = np.random.uniform() > 0.5
                if u:
                    initial[i] = helper_values.loc[pg_i, vn_i + 1]
                else:
                    initial[i] = helper_values.loc[pg_i, vn_i - 1]
            accu[i + 1] = initial

        line_df = pd.DataFrame(accu, columns=param_list)
        return [np.array(line_df.loc[i]) for i in range(line_df.shape[0])]

    morris_lines = [generate_one_line() for _ in range(r)]
    return morris_lines
