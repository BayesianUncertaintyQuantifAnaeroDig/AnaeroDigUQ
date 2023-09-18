"""
Submodule for permanent storage of variables.
Main functions are:
    Feed:
        load_dig_feed (from a csv file)
        feed_to_pd (conversion to panda.DataFrame for representation)
        save_dig_feed (save to csv file)
    Parameter:
        load_dig_param (from a json file)
        param_to_pd (conversion to panda.Series for representation)
        save_dig_param (save to json file)
    State:
        load_dig_state (from a json file)
        state_to_pd (conversion to panda.Series for representation)
        save_dig_state (save to json file)

"""

from typing import List

import numpy as np
import pandas as pd

from ._helper import (
    influent_state_col,
    initial_state_col,
    parameter_dict,
    predict_col,
)


# --------------- Feed ---------------
def load_dig_feed(path: str) -> np.ndarray:
    """Reads a feed object. Expects a .csv path"""
    feed = pd.read_csv(path)
    return feed[influent_state_col.keys()].to_numpy()


def feed_to_pd(feed) -> pd.DataFrame:
    return pd.DataFrame(
        feed, columns=influent_state_col.keys()  # Should raise an error if misshaped
    )


def save_dig_feed(feed: np.ndarray, path: str):
    feed_to_pd(feed).to_csv(path, index=False)


# --------------- Parameter ---------------
def load_dig_param(path: str) -> np.ndarray:
    """Reads a parameter. Expects a .json path"""
    param = pd.read_json(path, orient="index", typ="Series")
    return param[parameter_dict.keys()].to_numpy()


def param_to_pd(param: np.ndarray) -> pd.Series:
    return pd.Series(param, index=parameter_dict.keys())


def save_dig_param(param: np.ndarray, path: str) -> None:
    param_to_pd(param).to_json(path, orient="index")


# --------------- Digester State ---------------
def load_dig_state(path: str) -> np.ndarray:
    return pd.read_json(path, orient="index", typ="series")[
        initial_state_col.keys()
    ].to_numpy()


def state_to_pd(state: np.ndarray) -> pd.Series:
    return pd.Series(state, index=initial_state_col.keys())


def save_dig_state(state: np.ndarray, path: str):
    state_to_pd(state).to_json(path, orient="index")


# --------------- Digester states ---------------
def load_dig_states(path: str) -> np.ndarray:
    return pd.read_csv(path)[predict_col.keys()].to_numpy()


def states_to_pd(dig_states: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(dig_states, columns=predict_col.keys())


def save_dig_states(dig_states: np.ndarray, path: str) -> None:
    states_to_pd(dig_states).to_csv(path, index=False)


# --------------- Multiple digester states ---------------
def load_mult_states(path: str) -> List[np.ndarray]:
    """
    Opens a list of digester states as 3D array
    """
    data = np.loadtxt(path, delimiter=",")

    n_days = data.shape[1] // len(predict_col)
    return data.reshape((data.shape[0], n_days, len(predict_col)))


def save_mult_states(l_states: List[np.ndarray], path: str) -> None:
    """
    Save a list of digester states as a csv.
    """

    data = np.array(l_states)
    n_obj = len(data)
    np.savetxt(path, data.reshape((n_obj, np.prod(data.shape[1:]))), delimiter=",")
