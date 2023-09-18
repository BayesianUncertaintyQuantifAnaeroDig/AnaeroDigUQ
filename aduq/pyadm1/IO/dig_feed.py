"""
Class for Feed data

DigesterFeed class is inherited from numpy ndarray.
Can be loaded from a csv file using load_dig_feed.
Can be saved to a csv file using .save method.
"""

from typing import Union

import numpy as np
import pandas as pd

from ._helper_pd_np import influent_state_col, influent_to_numpy, influent_to_pandas


class DigesterFeed(np.ndarray):
    """
    Digester Feed (Influent) class. Inherited from numpy array.
    Format check on construction (shape).

    Added methods:
        to_pandas, which gives a view of the DigesterFeed as a pd.DataFrame
        save, which saves the DigesterFeed
        split, which splits a DigesterFeed object in two DigesterFeed objects
            containing feed information respectively before and after specified time

    """

    def __new__(cls, input_array, check_constraints=False):
        obj = np.asarray(input_array).view(cls)

        # Check shape of input_array
        if len(obj.shape) != 2:
            raise Exception("Incorrect number of dimensions for DigParam. Expected 2")
        if obj.shape[1] != len(influent_state_col):
            raise Exception(
                f"Incorrect number of columns for DigParam. Expected {len(influent_state_col)}"
            )
        if check_constraints:
            pass

        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

    def to_pandas(self):
        """Convert DigesterFeed object in a pandas.DataFrame"""
        return influent_to_pandas(self)

    def save(self, path):
        """Save DigesterFeed object to a .csv file"""
        self.to_pandas().to_csv(path, index=False)

    def split(self, time_split: float):
        """
        Returns a tuple containing the feed information up to time and the feed information after time.
        Split is done so that the first information can give prediction up to time included.
        """
        time_feed = np.array(self[:, 0])
        cond = time_feed < time_split
        index_thresh = np.sum(cond)

        feed_before = self[0 : (index_thresh + 1)]
        feed_after = self[index_thresh:]
        return (feed_before, feed_after)

    def __repr__(self):
        return self.to_pandas().__repr__()

    def __str__(self):
        return self.to_pandas().__str__()


def load_dig_feed(path) -> DigesterFeed:
    """
    Loads a digester feed from a csv file.
    """
    return constr_dig_feed(pd.read_csv(path))


def constr_dig_feed(dig_feed: Union[np.ndarray, pd.DataFrame]):
    """
    Constructs a digester feed from either an array or a DataFrame
    """
    if isinstance(dig_feed, pd.DataFrame):
        dig_feed = influent_to_numpy(dig_feed)
    return DigesterFeed(dig_feed)
