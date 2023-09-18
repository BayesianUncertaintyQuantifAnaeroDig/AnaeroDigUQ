"""
Noise Data

Routines used to noise data for Calibration/UQ benchmark
"""

import warnings

import numpy as np

from .IO import DigesterFeed, DigesterState, DigesterStates


def noise_influent(influent_state: DigesterFeed, noise_lev: float) -> DigesterFeed:
    """
    Noise influent (except time) with log-uniform multiplicative factor.
    No side effect on input.

    Arguments:
        influent_state: DigesterFeed to noise
        noise_lev: noise level used
    """

    inf_state_loc = influent_state.copy()
    if noise_lev > 0:
        noise_lev = -noise_lev
        noise = np.reshape(
            np.exp(
                np.random.uniform(
                    (-noise_lev),
                    noise_lev,
                    influent_state.shape[0] * (influent_state.shape[1] - 1),
                )
            ),
            (influent_state.shape[0], influent_state.shape[1] - 1),
        )
        inf_state_loc[:, 1:] = inf_state_loc[:, 1:] * noise
    elif noise_lev < 0:
        warnings.warn(
            "noise level given is negative. Returning influent_state unchanged"
        )
    return inf_state_loc


def noise_obs(obs: DigesterStates, noise_lev: float) -> DigesterStates:
    """
    Noise DigesterStates object (except time) with log-uniform multiplicative factor.
    No side effect on obs.

    Arguments:
        obs: DigesterStates to noise
        noise_lev: noise level used
    """
    obs_loc = obs.copy()
    if noise_lev > 0:
        noise = np.reshape(
            np.exp(
                np.random.uniform(
                    (-noise_lev), noise_lev, obs.shape[0] * (obs.shape[1] - 1)
                )
            ),
            (obs.shape[0], obs.shape[1] - 1),
        )
        obs_loc[:, 1:] = obs_loc[:, 1:] * noise
    elif noise_lev < 0:
        warnings.warn("noise level given is negative. Returning obs unchanged")
    return obs_loc


def noise_init_state(init_state: DigesterState, noise_lev: float) -> DigesterState:
    """
    Noise DigesterState object (except time) with log-uniform multiplicative factor.
    No side effect on init_state.

    Arguments:
        init_state: DigesterState to noise
        noise_lev: noise level used
    """
    init_loc = init_state.copy()
    if noise_lev > 0:
        noise = np.exp(np.random.uniform((-noise_lev), noise_lev, len(init_loc) - 1))
        init_loc[1:] = init_loc[1:] * noise
    elif noise_lev < 0:
        warnings.warn("noise level given is negative. Returning init_state unchanged")
    return init_loc
