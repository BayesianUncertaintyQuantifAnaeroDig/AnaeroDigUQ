"""
Miscellanous functions used throughout the package

Functions:
    blab: wrap up for silent
    timedrun: TimeOut option for function call decorator
    interpretation: right composition decorator
    post_modif: left composition decorator
    safe_call: Evaluation of any function without failure (returns None if Exception occured)
    par_eval: Parallelisation switch for function evaluation
    num_der: numerical differentiation
    vectorize: function vectorization

Function implementations may change but input/output structure sholud remain stable.
"""

import signal
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Iterable, List, Optional, Union

import numpy as np
from multiprocess import Pool
from numpy.typing import ArrayLike


class ShapeError(ValueError):
    """Exception class when array shape is not as expected"""


def blab(silent: bool, *args, **kwargs) -> None:
    """
    Wrap up for print. If silents, does not print, else, prints.
    """
    if not silent:
        print(*args, **kwargs)


# From https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"block timedout after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def timedrun(max_time: int):
    def deco(function):
        def wrap(*args, **kwargs):
            with timeout(max_time):
                return function(*args, **kwargs)

        return wrap

    return deco


X1 = Any
X2 = Any
Y1 = Any
Y2 = Any


def interpretation(
    fun: Callable[[X1], X2]
) -> Callable[[Callable[[X2, Any], Y1]], Callable[[X1, Any], Y1]]:
    """
    Decorator for composition on the right
        Takes as argument f, then transforms a function g(x, ...) into g(fun(x), ...)
    """

    def deco(gfun: Callable[[X2, Any], Y1]) -> Callable[[X1, Any], Y1]:
        def wrapper(x, *args, **kwargs):
            return gfun(fun(x), *args, **kwargs)

        return wrapper

    return deco


def post_modif(
    fun: Callable[[Y1], Y2]
) -> Callable[[Callable[[Any], Y1]], Callable[[Any], Y2]]:
    """
    Decorator for composition on the left.
        Takes as argument f, then transforms a function g into f o g.
    """

    def deco(gfun: Callable[[Any], Y1]) -> Callable[[Any], Y2]:
        def wrapper(*args, **kwargs):
            return fun(gfun(*args, **kwargs))

        return wrapper

    return deco


class SafeCallWarning(Warning):
    """Warning for safe_call context. Enables sending a warning of failure inside a safe_call
    decorated function even if UserWarning is filtered as an exception."""


# For type hints
Input = Any
Output = Any


def safe_call(fun: Callable[[Input], Output]) -> Callable[[Input], Union[None, Output]]:
    """
    Decorator to evaluate a function safely. If function call fails, returns None.

    Decorated function can still fail IF SafeCallWarning is filtered as an error (which completely
    defeats SafeCallWarning purpose) inside fun.
    """

    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception as exc:  # pylint: disable=W0703
            warnings.warn(
                f"Evaluation failed with inputs {args}, {kwargs}: {exc}",
                category=SafeCallWarning,
            )
            return None

    return wrapper


def par_eval(
    fun: Callable[[Input], Output], xs: Iterable[Input], parallel: bool, **kwargs
) -> List[Output]:
    """
    Evaluation of a function on a list of values. If parallel is True,
    computations are parallelized using multiprocess.Pool . Else list
    comprehension is used.
    """
    if parallel:
        with Pool() as pool:  # pylint: disable=E1102
            out = pool.map(fun, xs, **kwargs)
    else:
        out = [fun(x) for x in xs]
    return out


def num_der(
    fun: Callable[[ArrayLike], ArrayLike],
    x0: ArrayLike,
    f0: ArrayLike = None,  # pylint: disable=W0613
    rel_step: Optional[float] = None,
    parallel: bool = True,
) -> np.ndarray:
    """
    Returns the Jacobian of a function
    If f : shape1 -> shape2,
    the output is of shape (shape1, shape2)

    Arguments:
        fun: the function to derivate
        x0: the point at which to derivate the function
        f0: the value of fun at x0 (is not used since 2 point approximation of the derivative is used)
        parallel: should the evaluations of fun be parallelized
    Output:
        The approximated jacobian of fun at x0 as a np.ndarray of shape (shape_x, shape_y)
    """

    shape_in = x0.shape
    x0 = np.array(x0).flatten()

    dim = np.prod(shape_in)
    loc_fun = interpretation(lambda x: x.reshape(shape_in))(fun)

    if rel_step is None:
        rel_step = (np.finfo(x0.dtype).eps) ** (1 / 3)

    to_evaluate = np.full((2 * dim, dim), x0)

    delta_x = np.maximum(1.0, x0) * rel_step
    add_matrix = np.diag(delta_x)
    to_evaluate[::2] = to_evaluate[::2] + add_matrix

    to_evaluate[1::2] = to_evaluate[1::2] - add_matrix

    evals = np.array(par_eval(loc_fun, to_evaluate, parallel=parallel))

    der = evals[::2] - evals[1::2]

    for i, d_x in enumerate(delta_x):
        der[i] = der[i] / (2 * d_x)
    shape_out = der[0].shape
    return der.reshape(shape_in + shape_out)


def safe_inverse_ps_matrix(matrix: np.ndarray, eps: float = 10**-6) -> np.ndarray:
    """
    For a matrix which is supposed to be symmetric, positive definite, computes the inverse.

    All eigenvalues < eps will be heavily perturbed.
    This function is still work in progress
    """

    return np.linalg.inv(0.5 * (matrix + matrix.T) + eps * np.eye(len(matrix)))


def vectorize(
    fun: Callable, input_shape, convert_input=True, parallel=True
) -> Callable:
    """For a function fun which takes as input np.ndarray of shape input_shape and outputs
    arrays of shape output_shape, outputs the vectorized function which takes as input np.ndarray
    of shape (pre_shape, input_shape) and outputs np.ndarrat of shape (pre_shape, output_shape)
    """
    d = len(input_shape)

    def new_fun(xs) -> np.ndarray:
        if convert_input:
            xs = np.array(xs)
        pre_shape = xs.shape[:-d]
        xs.reshape(
            (
                np.prod(
                    pre_shape,
                )
                + input_shape
            )
        )
        out = np.array(par_eval(fun, xs, parallel=parallel))
        out.reshape(pre_shape + out.shape[1:])
        return out

    return new_fun
