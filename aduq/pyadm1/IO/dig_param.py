"""
Classes for Digester Parameter and Free Digester Parameter.

Basically a DigesterParameter is thought of as an object which can be directly fed into the
run_adm1 routine, while a FreeDigesterParameter has to be transformed into a DigesterParameter
object before being fed to run_adm1. A bijection exists between the two objects.

Optimisation procedures and part of the uncertainty quantification procedures involves the
FreeDigesterParameter class rather than DigesterParameter, as a FreeDigesterParameter is a
point in R^p while a DigesterParameter is a point in R_+^p. As its name suggests, a FreeDigesterParameter
has no constraint and any value is theoretically valid.

The transform between the DigesterParameter and FreeDigesterParameter is a logarithm applied component wise,
with a shift such that the FreeDigesterParameter with 0 values amounts to the default DigesterParameter as found
in the litterature (see _normalisation_param file).


DigesterParameter class is inherited from numpy ndarray.
Can be loaded from a json file using load_dig_param.
Can be saved to a json file using .save method.

FreeDigesterParameter class is inherited from numpy ndarray
Can be loaded from a json file using load_free_dig_param.
Can be saved to a json file using .save method.


Function free_param transforms a DigesterParameter into a FreeDigesterParameter
Function free_to_param transforms a FreeDigesterParameter into a DigesterParameter (.to_dig_param method can also be used)

interp_param is a decorator, which transforms a function taking as input a DigesterParameter into a function taking as input a
FreeDigesterParameter.
"""

import warnings
from typing import Any, List, Type, Union

import numpy as np
import pandas as pd

from ...misc import interpretation
from ._helper_pd_np import param_to_numpy, param_to_pandas, parameter_dict
from ._normalisation_param import max_param, renorm_param


class DigesterParameter(np.ndarray):
    """
    Digester Parameter class. Inherited from numpy ndarray.
    Format check on construction (shape).

    A DigesterParameter can be loaded from a json file using load_dig_param function.

    Added methods:
        to_pandas, which gives a view of the DigesterParameter as a pd.Series
        save, which saves the DigesterParameter as a json file.
    """

    def __new__(cls, input_array, check_constraints=False):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        if obj.shape != (len(parameter_dict),):
            raise Exception(
                f"Incorrect shape for DigesterParameter.\nFound shape {obj.shape}\nExpected shape {(len(parameter_dict),)}"
            )
        if check_constraints:
            pass
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

    def to_pandas(self, copy=True) -> pd.Series:
        """
        Transforms a DigesterParameter object to an user-friendly pandas.Series object.
        Used for representation.
        """
        return param_to_pandas(self)

    def save(self, path: str):
        """
        Saves a DigesterParameter object to a JSON file.
        The saved file can be loaded back to a DigesterParameter object through load_dig_param function
        """
        self.to_pandas().to_json(path, orient="index")

    def __repr__(self):
        return self.to_pandas().__repr__()

    def __str__(self):
        return self.to_pandas().__str__()


def load_dig_param(path) -> DigesterParameter:
    """
    Loads a digester parameter from a JSON file.
    """
    param = pd.read_json(path, orient="index", typ="Series")
    try:
        return constr_dig_param(param)
    except:
        print(param)
        raise Exception("Loading failed")


def constr_dig_param(param: Union[np.ndarray, pd.Series]) -> DigesterParameter:
    """
    Constructs a digester parameter from either a numpy ndarray or a pandas Series
    """
    if isinstance(param, pd.Series):
        param = param_to_numpy(param)
    return DigesterParameter(param)


def impose_param_constraints(param: DigesterParameter) -> DigesterParameter:
    # Check that every parameter is positive:
    param_temp = np.array([val if val > 0 else 0 for val in param])

    return DigesterParameter(param_temp)


renorm_param = constr_dig_param(pd.Series(renorm_param))

max_param = DigesterParameter(pd.Series(max_param))


def bound_param(param: DigesterParameter) -> DigesterParameter:
    """Clips the parameter values"""
    return np.minimum(param, max_param)


class FreeDigesterParameter(np.ndarray):
    """Free Digester Parameter class
    Used as a convenience for probability distributions, optimisation and some uncertainty quantification routines.

    Transform applied to DigesterParameter to obtain FreeDigesterParameter is, component wise,

        $\theta^f_i = \log(\theta_i/\theta^{default}_i)$

    A FreeDigesterParameter object should NOT be fed as is to run_adm1 or other functions.
    Method to_dig_param can be used to convert it to a DigesterParameter.
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if obj.shape != (len(parameter_dict),):
            raise Exception(
                f"Incorrect shape for DigesterParameter.\nFound shape {obj.shape}\nExpected shape {(len(parameter_dict),)}"
            )
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

    def to_pandas(self, copy=True):
        return param_to_pandas(self)

    def save(self, path):
        self.to_pandas().to_json(path, orient="index")

    def to_dig_param(self) -> DigesterParameter:
        return free_to_param(self)

    def __repr__(self):
        return self.to_pandas().__repr__()

    def __str__(self):
        return self.to_pandas().__str__()


def load_free_dig_param(path):

    return constr_free_dig_param(pd.read_json(path, orient="index", typ="Series"))


def constr_free_dig_param(param: Union[np.ndarray, pd.Series]):
    """
    Constructs a digester parameter from either a numpy ndarray or a panda Series
    """
    if isinstance(param, pd.Series):
        param = param_to_numpy(param)
    return FreeDigesterParameter(param)


def free_to_param(param: Type[np.ndarray]) -> DigesterParameter:
    """

    Standard interpretation for parameters

    """

    vals = np.exp(param) * renorm_param

    return DigesterParameter(vals)


def free_param(param: DigesterParameter) -> FreeDigesterParameter:

    vals = np.log(np.array(param) / renorm_param)

    return FreeDigesterParameter(vals)


interp_param = interpretation(free_to_param)

