"""
Tests for the proba package
"""
import warnings

import aduq.proba as proba
import numpy as np

print("Testing 'proba._helper' functions")
from aduq.proba._helper import _get_pre_shape, _shape_info, prod

# Test prod
print("Testing 'proba._helper.prod'")


def check_prod(shape):
    out = prod(shape)
    if not isinstance(out, int):
        warnings.warn(f"prod function does not output int for {shape}")
    if out <= 0:
        warnings.warn(f"prod function resulted in negative value for {shape}")


[check_prod(x) for x in [(), (1,), (2, 3), (10**16, 10**3, 10**4)]]

# Test _shape_info
print("Testing 'proba._helper._shape_info'")
if _shape_info(2, None) != (2, (2,)):
    warnings.warn("_shape_info failure with dimension passed")

if _shape_info(None, (4, 5)) != (20, (4, 5)):
    warnings.warn("_shape_info failure with shape passed")

if _shape_info(8, (4, 5)) != (20, (4, 5)):
    warnings.warn("_shape_info failure with incoherent shape and dim passed")

if _shape_info(20, (4, 5)) != (20, (4, 5)):
    warnings.warn("_shape_info failure with coherent shape and dim passed")

try:
    _shape_info(None, None)
    warnings.warn("_shape_info did not raise an error when no information was passed")
except ValueError:
    # This is normal behavior
    pass

# Test _get_pre_shape
print("Testing 'proba._helper._get_pre_shape'")
if _get_pre_shape(np.zeros((3, 4, 5)), (5,)) != (3, 4):
    warnings.warn("_get_pre_shape failed with one dimensional shape looked for")
if _get_pre_shape(np.zeros((3, 4, 5)), (3, 4, 5)) != ():
    warnings.warn("_get_pre_shape failed with zero dimensional output looked for")
if _get_pre_shape(np.zeros(5), (5,)) != ():
    warnings.warn("_get_pre_shape failed with zero dimensional output looked for")
try:
    _get_pre_shape(np.zeros((3, 4, 5)), (8, 5))
    warnings.warn("_get_pre_shape did not fail with incompatible shapes passed")
except ValueError:
    # This is normal behavior
    pass

try:
    _get_pre_shape(np.zeros((3, 4, 5)), (8, 3, 4, 5))
    warnings.warn("_get_pre_shape did not fail with incompatible shapes passed")
except ValueError:
    # This is normal behavior
    pass


### Tests for proba
print("Testing proba.Proba class")
sample_size = 10**4

## Case when proba outputs np.ndarray
print("Constructing uniform distribution as Proba instance with 'np_out=True'")
# Construction of a proba object
def gen(n):
    return np.random.uniform(0, 1, (n, 1))


def log_dens(xs):
    pre_shape = _get_pre_shape(xs, (1,))
    index_0 = (xs < 0) | (xs > 1)
    out = np.zeros(xs.shape)
    out[index_0] = -np.inf
    return out.reshape(pre_shape)


unif = proba.Proba(gen, log_dens, sample_shape=(1,), np_out=True)

# Test for contraction
print("Testing 'contract' method")
unif_contract = unif.contract(0.2)

# Check sample
sample = unif_contract(sample_size)

max_sample = np.max(sample)
min_sample = np.min(sample)
if (max_sample > 0.2) or (min_sample < 0):
    warnings.warn("contract method failed for uniform")

max_sample_theoretical = np.exp(np.log(0.0001) * sample_size)
min_sample_theoretical = 1 - max_sample_theoretical
if (max_sample < (0.2 * max_sample_theoretical)) or (
    min_sample > (0.2 * min_sample_theoretical)
):
    warnings.warn(
        "contract method seems to have failed for uniform (proba of error: 0.0002)"
    )

# Check log_dens
log_densities = unif_contract.log_dens(np.array([-1.0, 0.1, 0.5]).reshape((3, 1)))
expected = np.array([-np.inf, np.log(5.0), -np.inf])
if ~np.all(
    np.isclose(log_densities, expected, rtol=10**-5, atol=10**-5, equal_nan=True)
):
    warnings.warn("contract method resulted in improper log_density function")
    print(log_densities, expected)

# Test for shift
print("Testing 'shift' method")
unif_shift = unif.shift(1)
sample = unif_shift(10**4)
max_sample = np.max(sample)
min_sample = np.min(sample)
if (np.max(sample) > 2) or (np.min(sample) < 1):
    warnings.warn("shift method failed for uniform")

if (max_sample < (1 + max_sample_theoretical)) or (
    min_sample > (1 + min_sample_theoretical)
):
    warnings.warn(
        "shift method seems to have failed for uniform (proba of error: 0.0002)"
    )

log_densities = unif_shift.log_dens(np.array([0.5, 1.2, 2.5]).reshape((3, 1)))
expected = np.array([-np.inf, 0.0, -np.inf])
if ~np.all(
    np.isclose(log_densities, expected, rtol=10**-5, atol=10**-5, equal_nan=True)
):
    warnings.warn("shift method resulted in improper log_density function")
    print(log_densities, expected)

# Test for lin_transform
unif_lin = unif.lin_transform(np.array([[0.2]]), np.array([0.5]))
