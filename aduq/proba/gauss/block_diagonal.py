"""
Subpackage for block diagonal gaussian map

These can be constructed in two fashions:
- using the subset method from gaussian
- using the map_tensorize function and eventually the transform method to repermute output

The first method is simpler to implement BUT might involve more computations when evaluating
attributes (log_dens_der, map) and methods (kl, grad_kl, grad_right_kl). This is due to inverting
larger matrices rather than block components.

A final option could be to implement a specific class if computation time is a major consideration
"""
from ..proba_map import ProbaMap
from .Gauss import GaussianMap


def prob_param_idx(groups: list[list[int]]) -> list[int]:
    """
    Prepare the set of indexes of a ProbaParam for GaussianMap which determine the block covariance
    structure defined by the groups passed.

    Input is a list of list of int (each list of int contains sample indices belonging in a same
        group)
    Output is a list of int.
    """
    # Check sample_groups is well formatted
    items = [i for group in groups for i in group]
    set_items = set(items)
    n_tot = len(set_items)

    if n_tot != len(items):
        raise ValueError("An index should be in only one group")

    if set_items != set(range(n_tot)):
        raise ValueError(f"All indexes between 0 and {n_tot} should belong to a group")

    # Prevent side effect
    groups = groups.copy()

    return [i for group in groups for i in _prob_param_idx_from_group(group, n_tot)]


def _prob_param_idx_from_group(group_idx: list[int], n_tot: int) -> list[int]:
    """Used by _prob_param_idx"""
    accu = group_idx  # Side effect on copied item group_idx passed
    while len(group_idx) > 0:
        k = group_idx.pop()
        accu.append(n_tot + n_tot * k + k)
        for j in group_idx:
            accu.append(n_tot + n_tot * k + j)
            accu.append(n_tot + n_tot * j + k)
    return accu


def BlockDiagonalGaussianMap(groups: list[list[int]]) -> ProbaMap:
    """
    Outputs a ProbaMap object for block diagonal distributions.

    Efficiency of the class to be assessed (notably inversions of
    matrices).
    """
    n_tot = len({i for group in groups for i in group})
    gmap = GaussianMap(sample_dim=n_tot)
    return gmap.subset(prob_param_idx(groups))
