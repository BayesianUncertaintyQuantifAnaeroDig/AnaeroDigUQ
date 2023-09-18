"""
Classes for Predictions/Observations of Digester Data as well as initial state.

DigesterState class is inherited from numpy ndarray.
Can be loaded from a json file using load_dig_state.
Can be saved to a json file using .save method.

DigesterStates class is inherited from numpy ndarray
Can be loaded from a csv file using load_dig_states
Can be saved to a csv file using .save method.

Multiple DigesterStates can be saved either as a single file using save_mult_states (and loaded
using load_mult_states), or into individuals file inside a folder using save_list_dig_states
"""

import os
import warnings
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd

from ._helper_pd_np import pred_col, small_predictions_rm_numb

col_names = list(pred_col.keys())

## ------------------- DigesterState class and related -------------------

# To simplify code, all features of DigesterStates are included in DigesterState.
# S_co2, S_nh4_ion, q_gas, q_ch4, p_ch4, p_co2, VS and VSR are not used for
# initialization and can be set to NaN


class DigesterState(np.ndarray):
    """
    Digester State class. Inherited from numpy ndarray.
    Format check on construction (shape).
    Size is fixed but unknown values are allowed.

    Added methods:
        to_pandas, which gives a view of the DigesterState as a pd.Series
        save, which saves the DigesterState
    """

    def __new__(cls, input_array, check_constraints=False):

        obj = np.asarray(input_array).view(cls)

        # Check obj dimension
        if len(obj.shape) != 1:
            raise Exception(
                "Incorrect number of dimensions for DigesterState. Expected 1"
            )
        if len(obj) != (len(col_names)):
            raise Exception(
                f"Incorrect number of columns for DigesterState. Expected {len(col_names)}"
            )

        if check_constraints:
            # Positivity constraints
            # Remove NaNs before check
            obj_loc = [i for i in obj[1:] if not np.isnan(i)]
            if any(obj_loc < 0):
                raise Exception("All observations are expected to be positive.")

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return ()

    def to_pandas(self):
        return pd.Series(self, col_names)

    def save(self, path):
        self.to_pandas().to_json(path, orient="index")

    def __repr__(self):
        return (self.to_pandas()).__repr__()

    def __str__(self):
        return self.to_pandas().__str__()


def load_dig_state(path) -> DigesterState:
    """Load a dig state from a json file.
    Keys are saved to help with compatibility if dimension order changes.
    """
    return constr_dig_state(
        pd.read_json(path, orient="index", typ="series"), copy=False
    )


def constr_dig_state(
    dig_state: Union[dict, pd.Series, np.ndarray], copy: bool = True
) -> DigesterState:
    """
    Tries interpreting dig_state as a DigesterState object.
    - If input is dict or pd.Series, dig_state is reorder so that the keys match
    the standard order. Missing keys are filled as NaN
    - For other classes, object is directly passed to DigesterState, which will
    interpret input as an array-like

    Arguments:
        - dig_state, the object to be converted to DigesterState class.
        - copy, stating whether the output should be a copy of dig_state or if
        shared memory is allowed (default is True). Note that copy = False
        does not guarantee that output shares the input memory (not the case
        if dig_state is not a subclass of np.ndarray, for instance a list)
    Output:
        A DigesterState object if dig_state could be interpreted as one.

    Remark: when the DigesterState is constructed, the object inputed is translated
    to an array with asarray, so conversions to numpy is done at this stage.
    """
    if copy:
        dig_state_loc = dig_state.copy()
    else:
        dig_state_loc = dig_state

    if isinstance(dig_state_loc, DigesterState):
        return dig_state_loc

    if isinstance(dig_state_loc, dict):
        dig_state_loc = pd.Series(dig_state_loc)

    if isinstance(dig_state_loc, pd.Series):
        # Add missing keys
        to_add = set(col_names).difference(set(dig_state_loc.index))
        for missing in to_add:
            dig_state_loc[missing] = np.nan

        dig_state_loc = dig_state_loc[col_names]

    return DigesterState(dig_state_loc)


## ------------------- DigesterStates class and related -------------------


class DigesterStates(np.ndarray):
    """
    Digester States class. Inherited from numpy array.
    Format check on construction (shape).
    Reorder time at construction.

    Taking a subsample of rows of a DigesterStates object still outputs a DigesterStates object.
    Taking a subsample of columns of a DigesterStates object will not raise an error, but this
    should be avoided (or use asarray to avoid improperly using the class).

    Added methods:
        to_pandas, which gives a view of the DigesterStates as a pd.DataFrame
        save, which saves the DigesterStates
        find_state, which gives the state at a given time (or closest to the given time)
        split, which splits the DigesterStates in two digester states (before or equal given time, after given time)
    """

    def __new__(cls, input_array, check_constraints=False):

        obj = np.asarray(input_array).view(cls)

        # Check obj dimension
        if len(obj.shape) != 2:
            raise Exception(
                "Incorrect number of dimensions for DigesterStates. Expected 2"
            )
        if obj.shape[1] != (len(col_names)):
            raise Exception(
                f"Incorrect number of columns for DigesterStates.\nFound: {obj.shape[1]}\nExpected: {len(col_names)}"
            )

        time = obj[:, 0]
        if any(time[:-1] >= time[1:]):
            warnings.warn("Time is not ordered. Reordering the digester states.")
            sort_keys = np.argsort(time)
            obj = obj[sort_keys]

        if check_constraints:
            # Positivity constraints
            # NaNs are removed before check
            obj_loc = obj[:, 1:].flatten()
            obj_loc = [i for i in obj_loc if not np.isnan(i)]
            if any(obj_loc < 0):
                raise Exception("All observations are expected to be positive.")

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def split(self, time_split: float):
        """
        Split the DigesterStates object on time.
        If there is information for time time_split, stored in the first output.
        """
        time = self[:, 0]
        return (self[time <= time_split], self[time > time_split])

    def mask(self, inplace=True):
        """
        Remove part of the information in DigesterStates.
        Non NaNs columns after mask are time, S_va, S_bu, S_pro, S_ac, S_IN, q_gas, q_ch4, p_ch4, p_co2.
        """
        nan_cols = list(np.array(small_predictions_rm_numb))
        if inplace:
            self[:, nan_cols] = np.NaN
            return None

        new = self.copy()
        new[:, nan_cols] = np.NaN
        return new

    def time(self) -> np.ndarray:
        return np.array(self[:, 0]).copy()

    def last_state(self, copy=True) -> DigesterState:
        return constr_dig_state(self[self.shape[1], :], copy=copy)

    def init_state(self, copy=True) -> DigesterState:
        return constr_dig_state(self[0, :], copy=copy)

    def find_state(self, time, tol=1, raise_error=True, copy=True) -> DigesterState:
        """
        Given a time, returns the state closest to that time as a pd.Series,
        if the time difference is less than tol, else raise an exception.

        Args:
            time, the time for which we want to extract the state
            tolerance, the maximum time tolerance (in days) accepted between time and times
            in the digester information. If no time stamp in digester information is
            smaller than tolerance, then either raise an exception or sends a warning
        Outputs:
            A pd.Series describing the digester state at time.

        """
        time_vect = self[:, 0]
        delta_times = np.absolute(time - time_vect)
        k = np.argmin(delta_times)

        if delta_times[k] > tol:
            err_message = (
                f"No digester state information between {time - tol} and {time + tol}."
            )
            if raise_error:
                raise Exception(err_message)
            else:
                warnings.warn(err_message)
                return None

        return constr_dig_state(pd.Series(self[k], col_names), copy=copy)

    def to_pandas(self, copy=False):
        if copy:
            return pd.DataFrame(self, columns=col_names).copy()
        else:
            return pd.DataFrame(self, columns=col_names)

    def save(self, path):
        self.to_pandas().to_csv(path, index=False)

    def __repr__(self):
        return self.to_pandas().__repr__()

    def __str__(self):
        return self.to_pandas().__str__()


def load_dig_states(path) -> DigesterStates:
    return constr_dig_states(pd.read_csv(path), copy=False)


def constr_dig_states(
    dig_states: Union[np.ndarray, pd.DataFrame, dict], copy=True
) -> DigesterStates:
    """
    Constructs digester states from either an np.ndarray like, pd.DataFrame or dict.
    If dict, it is assumed that it can be translated as a pd.DataFrame (i.e., formatted
    as state:state_values_array_like)

    Arguments:
        - dig_states, the object to be converted to DigesterStates
        - copy, stating whether the output should be a copy of dig_states or if
        shared memory is allowed (default is True). Note that copy = False
        does not guarantee that output shares the input memory (not the case
        if dig_states is not a subclass of np.ndarray, for instance a dict or list of list)
    Output:
        A DigesterStates object if dig_state could be interpreted as one.

    Remark: when the DigesterStates is constructed, the object inputted is translated
    to an array with asarray, so conversions to numpy is done at this stage.
    """
    if copy:
        dig_states_loc = dig_states.copy()
    else:
        dig_states_loc = dig_states

    if isinstance(dig_states_loc, dict):
        dig_states_loc = pd.DataFrame(dig_states_loc)
    if isinstance(dig_states_loc, pd.DataFrame):
        to_add = set(col_names).difference(set(dig_states.columns))
        for missing in to_add:
            dig_states[missing] = np.nan
        dig_states = dig_states[col_names]
    return DigesterStates(dig_states)


def time_align(
    x: DigesterStates, y: DigesterStates
) -> Tuple[DigesterStates, DigesterStates]:
    """
    Returns subsetted versions of x and y such that the time index are identical.
    In this version, we assume that the time index of x and y are sequences of consecutives integers
    """
    x_time = x[:, 0]
    y_time = y[:, 0]
    min_x_y = np.max(x_time.min(), y_time.min())
    max_x_y = np.min(x_time.max(), y_time.max())
    return (
        x[(x_time >= min_x_y) & (x_time <= max_x_y)],
        y[(y_time >= min_x_y) & (y_time <= max_x_y)],
    )


# Multiple digester states
def load_mult_states(path: str) -> List[DigesterStates]:
    """
    Opens a list of digester states as 3D array. All the digester states share the same shape.
    """
    data = np.loadtxt(path, delimiter=",")

    n_days = data.shape[1] // len(pred_col)
    data = data.reshape((data.shape[0], n_days, len(pred_col)))
    return [DigesterStates(x) for x in data]


def save_mult_states(l_states: List[DigesterStates], path: str) -> None:
    """
    Save a list of digester states (or 3D array seen as an iterable of digester states) with identical shape
    as a unique csv file, intended to be loaded using load_mult_states function.

    Every digester states object must have the same shape!

    Ex:
    import numpy as np
    # Create thrash digester states information
    n_days = 10
    mult_dig_states = [DigesterStates(np.random.uniform(0,1, (n_days, len(pred_col))))]

    # Save data
    save_mult_states("data.csv")

    # reload data
    from of import listdir
    loaded_data = load_mult_states("data.csv")
    """

    data = np.array(l_states)
    n_obj = len(data)
    np.savetxt(path, data.reshape((n_obj, np.prod(data.shape[1:]))), delimiter=",")


def save_list_dig_states(xs: Iterable[DigesterStates], path: str) -> None:
    """
    Saves a list of digester states as individual files in a folder, meant to be iteratively
    loaded by load_dig_states function.

    Ex:
    import numpy as np
    # Create thrash digester states information
    n_days = 10
    mult_dig_states = [DigesterStates(np.random.uniform(0,1, (n_days, len(pred_col))))]

    # Save data
    save_list_dig_states("my_dir")

    # reload data
    from of import listdir
    loaded_data = [load_dig_states(file) for file in listdir("my_dir")]
    """
    index = 0
    for dig_states in xs:
        dig_states.save(os.path.join(path, "dig_states_" + str(index) + ".csv"))
        index += 1
