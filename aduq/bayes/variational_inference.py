r"""
Classes and functions for PAC Bayesian optimisation problem

VI objective is Catoni's bound (see https://doi.org/10.48550/arXiv.2110.11216)

    $$\int f(x) p(\theta, dx) + temp * KL(p(\theta), p0)$$

Implementation is in line with packages proba and optim.
- The variational class is coded as ProbaMap
- The prior p0 is described as a distribution parameter.
- The output is inherited from OptimResult object

It is assumed that the bottleneck of the procedure is the function evaluation and not data
generation from either $p(\theta)$ or $p0$. As such, ways to recycle function calls along the
optimization procedure are explored. The first consists in estimating the integral using samples
generated not only from $p(\theta)$ through weight correction. The second in constructing a proxy
through KNN (other methods could be implemented) using all or most of the samples.
"""

import warnings
from typing import Callable, Iterable, Optional, Tuple, Type, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm
from sklearn.neighbors import KNeighborsRegressor  # pylint: disable=E0401

from ..misc import ShapeError, blab, par_eval
from ..optim import OptimResult
from ..proba import Proba, ProbaMap, ProbaParam, SamplePoint


# Warnings and exception
class ProbBadGrad(Warning):
    """
    Warning class to indicate that a badly estimated gradient step has occured.
    """


class FullMemory(Exception):
    """Custom Error raised when trying to store memory to an already full memory manager"""


# Evaluate log_dens on a list of parameters
def _half_log_dens(sample: Iterable[SamplePoint], distrib: Type[Proba]):
    r"""
    Evaluate the log density of a distribution at each point in a sample

    Args:
        sample, a list of SamplePoint
        distrib, the distribution whose log_dens is evaluated

    Meant to be used as weight correction, the full correction being:
        $\exp( _half_log_dens(sample, distrib1) - _half_log_dens(sample, distrib0))$
    if the sample is generated through distrib0, hence the name.
    """
    return np.array(distrib.log_dens(sample))


class HistVILog:
    r"""
    Manages the high level history of a PAC Bayesian optimisation problem of form

    $$S_{VI}(\theta) = E_{p(\theta)}[score] + C kl(p(\theta), p0).$$
    where $E_{p(\theta)}[score]$ is the expected value (or mean) of the score of the probability
    distribution $p(\theta)$.

    Stored data can be accessed through methods:
        distr_pars ($\theta$),
        VI_scores ($S_{VI}(\theta)$),
        KLs ($kl(p(\theta), p0)$),
        means ($E_{p(\theta)}[score]$)
    which take as input a number of data (optional, if None returns all data)

    Data is stored in np.ndarray of fixed sized defined at initialisation. These should never be
    accessed by the users as they can be misleading (data suppression is lazy, i.e. only a count
    of stored data is changed)

    The class is initialised by:
        A ProbaMap object (the function p mapping $\theta$ to a distribution)
        The maximal number of elements stored.
    """

    def __init__(self, distr_map: Type[ProbaMap], n: int):

        self.distr_map = distr_map

        # Prepare memory
        self._distr_pars = np.zeros((n,) + distr_map.distr_param_shape)
        self._VI_scores = np.zeros(n)
        self._KLs = np.zeros(n)
        self._means = np.zeros(n)

        # Specify memory size and amount filled
        self.n_filled = 0
        self.n = n

    def is_empty(self) -> bool:
        """Checks if the history is empty"""
        return self.n_filled == 0

    def _full(self) -> bool:

        """Checks if the history is full"""

        return self.n == self.n_filled

    def add(
        self,
        distr_pars: list[ProbaParam],
        scores: list[float],
        KLs: list[float],
        means: list[float],
    ) -> None:
        """
        Store multiple new information in the history
        """
        n = len(distr_pars)

        if not (n == len(scores)) & (n == len(KLs) & (n == len(means))):
            raise ValueError(
                "distr_pars, scores, KLS and means should have same length"
            )

        distr_pars = np.array(distr_pars)

        if not (distr_pars.shape == ((n,) + self.distr_map.distr_param_shape)):
            raise ShapeError(
                f"distr_pars field should be an array like of shape (N,) + {self.distr_map.distr_param_shape}"
            )

        n0 = self.n_filled

        if self._full():
            raise FullMemory("Already full")
        if n + n0 > self.n:
            raise Warning(f"Too much data is passed. Only storing first {self.n - n0}.")

        n = min(n, self.n - n0)

        self._distr_pars[n0 : (n0 + n)] = distr_pars
        self._VI_scores[n0 : (n0 + n)] = scores
        self._KLs[n0 : (n0 + n)] = KLs
        self._means[n0 : (n0 + n)] = means

        self.n_filled = self.n_filled + n

    def add1(self, distr_par: ProbaParam, score: float, KL: float, mean: float) -> None:
        """
        Store new information in the history. Similar to add, but does not expect list like elements.
        """
        if self._full():
            raise FullMemory("Already full")

        n = self.n_filled
        self._distr_pars[n] = distr_par
        self._VI_scores[n] = score
        self._KLs[n] = KL
        self._means[n] = mean

        self.n_filled += 1

    def get(self, k: int = 1) -> tuple[np.ndarray]:
        """
        Outputs the description of the last k elements added to the memory
        """
        return self.distr_pars(k), self.VI_scores(k), self.KLs(k), self.means(k)

    def suppr(self, k: int = 1) -> None:
        """
        To all purposes, deletes the k last inputs and returns the deleted inputs.
        """
        self.n_filled = max(0, self.n_filled - k)
        return self.distr_pars(k), self.VI_scores(k), self.KLs(k), self.means(k)

    def distr_pars(self, k: Optional[int] = None) -> np.ndarray:
        """
        Outputs the last k distribution parameters.
        Last element is last distribution parameter
        """

        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)

        return self._distr_pars[init : self.n_filled]

    def VI_scores(self, k: Optional[int] = None) -> np.ndarray:
        """
        Outputs the last VI scores (last element is last score)
        """

        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)

        return self._VI_scores[init : self.n_filled]

    def KLs(self, k: Optional[int] = None) -> np.ndarray:
        """
        Outputs the last k KLs (last element is last KL)
        """

        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)

        return self._KLs[init : self.n_filled]

    def means(self, k: Optional[int] = None) -> np.ndarray:
        """
        Outputs the last k means (last element is last mean)
        """

        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)

        return self._means[init : self.n_filled]

    def best(self) -> Tuple[np.ndarray, float]:

        if self.n_filled == 0:
            raise ValueError("Empty history")

        pars, scores = self.distr_pars(), self.VI_scores()

        best_ind = np.nanargmin(scores)
        return pars[best_ind], scores[best_ind]


class AccuSampleVal:
    """
    Manages the low level history of a PAC Bayesian optimisation problem.

    Data can be accessed through methods
        sample (all SamplePoints generated),
        vals (the score of each  SamplePoint),
        gen_tracker (when was each  SamplePoint generated)
    which take as input a number of data (optional, if None returns all data)

    sample is a list of points x,
    vals the list of evaluations of the scoring function at x,
    half_ldens the half log densities for x wrt to the distribution from which x was generated,
    gen_tracker the information pointing from which distribution x was generated (latest generation is 0,
        -1 indicates that the sample point is not yet generated)

    Data is stored in np.ndarray of fixed sized defined at initialisation. These should never be
    accessed by the users as they can be misleading (data suppression is lazy, i.e. only a count of
    stored data is changed).

    It is possible to increase memory size through extend_memory method.
    """

    def __init__(self, sample_shape: tuple[int], n_tot: int):

        self.sample_shape = sample_shape

        self._sample = np.zeros((n_tot,) + sample_shape)
        self._vals = np.zeros(n_tot)
        self._gen_tracker = np.full(n_tot, -1)

        self.n_gen = 0

        self.n_filled = 0
        self.n_tot = n_tot

    def extend_memory(self, n_add: int) -> None:
        n_tot = self.n_tot + n_add
        n_filled = self.n_filled

        sample = np.zeros((n_tot,) + self.sample_shape)
        vals = np.zeros(n_tot)
        gen_tracker = np.full(n_tot, -1)

        sample[:n_filled] = self.sample()
        vals[:n_filled] = self.vals()
        gen_tracker[:n_filled] = self.gen_tracker()

        self._sample = sample
        self._vals = vals
        self._gen_tracker = gen_tracker

        self.n_tot = n_tot

    def add(self, sample: ArrayLike, vals: ArrayLike) -> None:
        """
        Add a new generation to memory.
        """
        sample = np.array(sample)
        m = len(sample)

        if sample.shape != (m,) + self.sample_shape:
            raise ShapeError(
                f"Expected shape {(m,) + self.sample_shape} for shape, got {sample.shape}"
            )

        n = self.n_filled

        if (n + m) > self.n_tot:
            warnings.warn("Maximum number of data reached")
            m = self.n_tot - n

        self._sample[n : (n + m)] = sample[:m]
        self._vals[n : (n + m)] = vals[:m]

        self._gen_tracker[: (n + m)] += 1

        self.n_gen += 1
        self.n_filled = n + m

    def suppr(self, K):
        """Deletes the last K generations"""
        gen_tracker = self._gen_tracker.copy()
        gen_tracker = np.clip(gen_tracker - K, a_min=-1, a_max=None)

        self.n_gen = max(0, self.n_gen - K)
        self.n_filled = np.sum(gen_tracker >= 0)
        self._gen_tracker = gen_tracker

    def sample(self, k: Optional[int] = None):
        """Clean look at the sample"""
        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)

        return self._sample[init : self.n_filled]

    def vals(self, k: Optional[int] = None):
        """Clean look at the sample evaluations"""

        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)

        return self._vals[init : self.n_filled]

    def gen_tracker(self, k: Optional[int] = None):
        """Clean look at the sample generations"""

        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)

        return self._gen_tracker[init : self.n_filled]

    def knn(self, k, *args, **kwargs):
        """
        Future:
            Using KNeighborsRegressor.score could be useful to choose which values should be
                evaluated
        """

        knn = KNeighborsRegressor(*args, **kwargs)
        knn.fit(self.sample(k), self.vals(k))

        return knn.predict

    def __repr__(self):
        return f"AccuSampleVal object with {self.n_filled} / {self.n_tot} objects"


class AccuSampleValDens(AccuSampleVal):
    """
    Manages the low level history of a PAC Bayesian optimisation problem
    Inherited from AccuSampleVal class (added half_ldens information)

    Data can be accessed through methods
        sample,
        half_ldens,
        vals,
        gen_tracker
    which take as input a number of data (optional, if None returns all data)

    sample is a list of points x,
    vals the list of evaluations of the scoring function at x,
    half_ldens the half log densities for x wrt to the distribution from which x was generated,
    gen_tracker the information pointing from which distribution x was generated (latest generation is 0,
        -1 indicates that the sample point is not yet generated)

    Data is stored in np.ndarray of fixed sized defined at initialisation. These should never be
    accessed by the users as they can be misleading (data suppression is lazy, i.e. only a count of
    stored data is changed).

    Main method is grad_score, used for the PAC Bayesian optimisation problem with corrected weights.

    Note:
        Half log density information is used to efficiently recompute the density ratio with an unknown
        distribution.
    """

    def __init__(self, sample_shape: tuple, n_tot: int):
        super().__init__(sample_shape, n_tot)

        self._half_ldens = np.zeros(n_tot)

    def add(self, sample, half_ldens, vals) -> None:  # pylint: disable=W0221
        # Format input
        sample = np.array(sample)
        m = len(sample)

        # Check conforming inputs
        if sample.shape != (m,) + self.sample_shape:
            raise ShapeError(
                f"Expected shape {(m,) + self.sample_shape} for shape, got {sample.shape}"
            )

        if (len(half_ldens) != m) or (len(vals) != m):
            raise ValueError("sample, hald_ldens and vals should have same length")
        n = self.n_filled

        # Check that full memory is not exceeded
        if (n + m) > self.n_tot:
            warnings.warn("Maximum number of data reached")
            m = self.n_tot - n

        # Add information to memory
        self._sample[n : (n + m)] = sample[:m]
        self._half_ldens[n : (n + m)] = half_ldens[:m]
        self._vals[n : (n + m)] = vals[:m]

        # Update generations
        self._gen_tracker[: (n + m)] += 1

        # Update filled memory size
        self.n_gen += 1
        self.n_filled = n + m

    def extend_memory(self, n_add: int) -> None:
        AccuSampleVal.extend_memory(self, n_add)
        half_ldens = np.zeros(self.n_tot)
        half_ldens[: self.n_filled] = self.half_ldens()
        self._half_ldens = half_ldens

    def half_ldens(self, k: Optional[int] = None) -> np.ndarray:
        """Clean look at the half log densities"""
        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)

        return self._half_ldens[init : self.n_filled]

    def corr_weights(
        self,
        distrib: Proba,
        k: Optional[int] = None,
    ) -> tuple[np.ndarray]:
        r"""
        Selects the k last parameters and return them along with the evaluations and the correct
        weight corrections.

        The resulting samples and weights can be used to estimate integrals through
            $$\mathbb{E}_{distrib}[f(x)] \simeq 1/N \sum \omega_i f(x_i)$$
        This is integral estimation is unbiased (variance analysis is not straightforward). The sub
        sums for each generation are also unbiased (but they are correlated with one another).

        TO DO:
            - For generation 0 (current), weights correction are computed by reevaluating the log
              densities. These are already known (basically no need for any loss correction).
              This could be modified not to have to reevaluate log_dens.
        """
        if k is None:
            k = self.n_filled

        return (
            self.sample(k),
            self.vals(k),
            _half_log_dens(self.sample(k), distrib) - self.half_ldens(k),
        )

    def grad_score(
        self,
        d_map: Type[ProbaMap],
        param: np.ndarray,
        gen_weights: Optional[Union[list, dict]] = None,
        gen_decay: float = 0.0,
        k: Optional[int] = None,
    ) -> tuple[np.ndarray, float, float]:

        r"""
        Outputs the derivative and evaluation at param of

        $$J(param) = \sum_{g>0} J_g(param) \exp(- g * gen_decay) / \sum_{g>0} \exp(-g * gen_decay)$$

        where J_g uses the sample S_g from generation g generated from param_g to estimate the mean through

        J_g(param) =  \sum_{x \in S_g} score(x) * \exp(log_dens(x, param) - log_dens(x, param_g)) / \lvert S_g \rvert

        The intuition being that if the distributions generating all parameters are similar, then it is beneficial to
        use the previous evaluations of the score function in order to minimize the variance of the derivative estimate.

        Note:
            if log_dens(x, param) - log_dens(x, param_g) is too high (
            i.e. the point x generated through distribution param_g is deemed much more likely to have been generated from
            param than param_g
            ), then problematic behaviour might happen, the impact of this single point becoming disproportionate.

        Args:
            d_map, the distribution map used in the PAC Bayesian optimisation problem
            param, the parameter at which the derivative is to be computed
            gen_weights, an optional list of weights specifying how each generation should be weighted (first element = latest generation)
            gen_decay, used if gen_weights is None.
                Controls speed of exponentially decaying given to generation k through
                    w_k = exp(-gen_decay * k).
                Default is 0 (no decay, all generation with same weight).
            k, controls maximum number of sample used. None amounts to all sample used.
        """
        # Construct current distribution
        distrib = d_map(param)
        # Prepare log_dens_der function
        der_log = d_map.log_dens_der(param)

        # Obtain proper weight corrections for samples from previous generations
        sample, vals, log_dens = self.corr_weights(distrib, k=k)

        # Set up weight given to a generation
        n_gen = self.n_gen
        if gen_weights is None:
            gen_weights = [np.exp(-gen_decay * i) for i in range(n_gen)]

        # Tackle case where gen_weights information passed is insufficient
        if len(gen_weights) < n_gen:
            warnings.warn(
                f"Missing information in gen_weights. Giving weight 0 to all generations further than {len(gen_weights)}"
            )
            gen_weights = list(gen_weights) + [
                0 for i in range(n_gen - len(gen_weights))
            ]

        # Prepare specific weight given to each sample
        gen_tracker = self.gen_tracker(k)
        count_per_gen = [np.sum(gen_tracker == i) for i in range(n_gen)]

        gen_factor = np.array(
            [gen_weights[gen] / count_per_gen[gen] for gen in gen_tracker]
        )
        gen_factor = gen_factor / np.sum(gen_factor)

        weights = np.exp(log_dens) * gen_factor
        weights = weights / np.sum(weights)

        # Compute mean value
        mean_val = np.sum(vals * weights)
        # Compute uncertainty using last generation only
        UQ_val0 = np.std(vals[gen_tracker == 0]) / np.sqrt(np.sum(gen_tracker == 0) - 2)

        # Compute estimation of mean score gradient
        grads = der_log(sample)
        grad = np.tensordot((vals - mean_val) * weights, grads, (0, 0))

        return grad, mean_val, UQ_val0

    def __repr__(self):
        return f"AccuSampleValDens object with {self.n_filled} / {self.n_tot} objects"


def gen_eval_samp(
    distrib: Proba,
    func: Callable[[SamplePoint], Union[float, np.ndarray]],
    n: int,
    parallel: bool = True,
    vectorized: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    sample = distrib(n)
    l_dens = _half_log_dens(sample, distrib)
    if vectorized:
        vals = func(sample)
    else:
        vals = par_eval(func, sample, parallel=parallel)
    return (sample, l_dens, vals)


class OptimResultVI(OptimResult):
    """
    Inherited from OptimResult.

    Added fields:
    - end_param
    - log_vi
    - sample_val
    """

    def __init__(
        self,
        opti_param: ProbaParam,
        converged: bool,
        opti_score: float,
        hist_param: list[ProbaParam],
        hist_score: list[float],
        end_param: ProbaParam,
        log_vi: HistVILog,
        bin_log_vi: HistVILog,
        sample_val: AccuSampleValDens,
    ):
        super().__init__(opti_param, converged, opti_score, hist_param, hist_score)
        self.log_vi = log_vi
        self.bin_log_vi = bin_log_vi
        self.sample_val = sample_val
        self.end_param = end_param


def optim_VI_wc(
    fun: Callable[[SamplePoint], float],
    distr_map: Type[ProbaMap],
    temperature: float = 1.0,
    prior_param: Optional[ProbaParam] = None,
    post_param: Optional[ProbaParam] = None,
    prev_eval: Optional[AccuSampleValDens] = None,
    index_train: Optional[list[int]] = None,
    eta: float = 0.05,
    chain_length: int = 10,
    per_step: int = 100,
    xtol: float = 10**-8,
    k: Optional[int] = None,
    gen_decay: float = 0.0,
    momentum: float = 0.0,
    refuse_conf: float = 0.99,
    corr_eta: float = 0.5,
    parallel: bool = True,
    vectorized: bool = False,
    print_rec: int = 1,
    silent: bool = False,
) -> OptimResultVI:
    """
    Variational inference Optimisation procedure.

    Function calls recycling is achieved through reuse with weight correction.

    If vectorized, it is assumed that the score function is vectorized (i.e one can directly
    evaluate multiple scores using fun(samples))
    """
    # Interpret setting
    refuse_factor = norm.ppf(refuse_conf)

    eta = eta * (1 - momentum)

    if prior_param is None:
        prior_param = distr_map.ref_param.copy()

    if post_param is None:
        post_param = prior_param.copy()

    ini_post_param = post_param.copy()

    n_param = np.prod(prior_param.shape)
    if index_train is None:
        index_no_train = []
        index_train = list(range(n_param))
    else:
        index_no_train = list(set(range(n_param)).difference(index_train))
        index_train = list(index_train)

    distr_param_shape = distr_map.distr_param_shape
    # Initialise
    log_vi = HistVILog(distr_map, n=chain_length + 1)
    bin_log_vi = HistVILog(distr_map, n=chain_length)

    if prev_eval is None:
        acc = AccuSampleValDens(distr_map.sample_shape, per_step * chain_length)
    else:
        acc = prev_eval
        acc.extend_memory(per_step * chain_length)

    distr = distr_map(post_param)

    grad_KL = distr_map.grad_kl(prior_param)

    v = np.zeros(distr_map.distr_param_shape)

    prev_score = np.inf

    i = 0
    converged = False
    while (i < chain_length) & (not converged):
        # Generate and evaluate new samples
        acc.add(
            *gen_eval_samp(
                distr, fun, per_step, parallel=parallel, vectorized=vectorized
            )
        )

        # Move parameter accordingly
        der_score, m_score, UQ_score = acc.grad_score(
            distr_map, post_param, k=k, gen_decay=gen_decay
        )
        der_KL, KL = grad_KL(post_param)

        score_VI = m_score + temperature * KL

        # Bad step removal mechanism
        # In case where a step meaningfully harms the score,
        # go back one step, re initialize speed, change step size
        if score_VI - prev_score > refuse_factor * UQ_score:
            warnings.warn(
                f"""
            Harmful step removed.
            (Previous score: {prev_score}, new_score: {score_VI}, UQ: {UQ_score}))""",
                category=ProbBadGrad,
            )

            v = 0
            eta = eta * corr_eta

            acc.suppr(2)
            bin_log_vi.add(
                *log_vi.suppr(1)
            )  # Go back one generation and store deleted information

            try:
                post_param = log_vi.distr_pars()[-1]
                distr = distr_map(post_param)
                prev_score = log_vi.VI_scores()[-1]
            except IndexError:
                # If the above fails, means back to the beginning (no previous data logged)
                # Set previous score to infinity
                post_param = ini_post_param.copy()
                distr = distr_map(post_param)
                prev_score = np.inf
            converged = False

        else:

            log_vi.add1(distr_par=post_param, score=score_VI, KL=KL, mean=m_score)

            v_new = -eta * (der_score + temperature * der_KL)
            v_new_flat = v_new.flatten()
            v_new_flat[index_no_train] = 0.0
            v_new = v_new_flat.reshape(distr_param_shape)
            v = v_new + v * momentum

            post_param = post_param + v

            prev_score = score_VI
            score_VI = m_score + temperature * KL
            if (i % print_rec) == 0:
                blab(
                    silent, f"Score at step {i}: {score_VI} (KL: {KL}, score:{m_score})"
                )
            distr = distr_map(post_param)

            converged = np.max(np.abs(v)) < xtol

        i += 1

    opt_param, opt_score = log_vi.best()

    return OptimResultVI(
        opti_param=opt_param,
        converged=converged,
        opti_score=opt_score,
        hist_param=log_vi.distr_pars(),
        hist_score=log_vi.VI_scores(),
        end_param=post_param,
        log_vi=log_vi,
        bin_log_vi=bin_log_vi,
        sample_val=acc,
    )


def optim_VI_knn(
    fun,
    distr_map: ProbaMap,
    prior_param: Optional[ProbaParam] = None,
    post_param: Optional[ProbaParam] = None,
    temperature: float = 1.0,
    prev_eval: Optional[AccuSampleVal] = None,
    index_train: Optional[list[int]] = None,
    eta: float = 0.05,
    chain_length: int = 10,
    per_step: int = 100,
    per_step_eval: int = 10000,
    xtol: float = 10**-8,
    k: Optional[int] = None,
    momentum: float = 0.0,
    # refuse_conf: float = 0.99,
    corr_eta: float = 0.5,
    parallel: bool = True,
    vectorized: bool = False,
    print_rec: int = 1,
    silent=False,
    **kwargs,
) -> OptimResultVI:
    """
    Variational inference Optimisation procedure.

    Function calls recycling is achieved by building proxy function with KNN.

    If vectorized, it is assumed that the score function is vectorized (i.e one can directly
    evaluate multiple scores using fun(samples))
    """
    ## Parameter interpetation/ set up

    # refuse_factor = norm.ppf(refuse_conf)
    # Correct actual speed for momentum
    eta = eta * (1 - momentum)

    # Set up distribution parameters
    if prior_param is None:
        prior_param = distr_map.ref_param.copy()

    if post_param is None:
        post_param = prior_param.copy()

    # keep track of initial post
    ini_post_param = post_param.copy()

    # Index train preparation
    n_param = np.prod(prior_param.shape)
    if index_train is None:
        index_no_train = []
        index_train = list(range(n_param))
    else:
        index_no_train = list(set(range(n_param)).difference(index_train))
        index_train = list(index_train)

    distr_param_shape = distr_map.distr_param_shape

    ## Initialise/ Prepare loop
    if prev_eval is None:
        acc = AccuSampleVal(distr_map.sample_shape, per_step * chain_length)
    else:
        acc = prev_eval
        acc.extend_memory(per_step * chain_length)

    log_vi = HistVILog(distr_map, n=chain_length + 1)
    bin_log_vi = HistVILog(distr_map, n=chain_length)

    distr = distr_map(post_param)
    grad_KL = distr_map.grad_kl(prior_param)

    v = np.zeros(distr_map.distr_param_shape)

    i = 0
    converged = False
    prev_score = np.inf

    # Main loop
    while (i < chain_length) & (not converged):
        # Generate/ evaluate new sample
        sample = distr(per_step)
        if vectorized:
            vals = fun(sample)
        else:
            vals = par_eval(fun, sample, parallel=parallel)

        # Store new sample
        acc.add(sample, vals)

        # Learn new interpolation
        interpol = acc.knn(k, **kwargs)
        der_log = distr_map.log_dens_der(post_param)

        # Generate/ evaluate gradient on new sample
        l_sample = distr(per_step_eval)
        l_vals = interpol(l_sample)
        l_grads_log = der_log(l_sample)

        mean_val = np.mean(l_vals)
        der_val = np.tensordot((l_vals - mean_val), l_grads_log, (0, 0)) / per_step_eval

        der_KL, KL = grad_KL(post_param)

        # Look at score of current parameter
        score_VI = mean_val + temperature * KL

        if score_VI > prev_score:
            # Refuse last step, go back to previous parameter/ forget speed

            blab(
                silent,
                f"Undo last step (last score {prev_score}, new score {score_VI})",
            )
            v = 0
            eta = eta * corr_eta

            # Collect information on removed parameter
            bin_log_vi.add(*log_vi.suppr(1))

            # Back to previous set up
            try:
                post_param = log_vi.distr_pars()[-1]
                distr = distr_map(post_param)
                prev_score = log_vi.VI_scores()[-1]
            except IndexError:
                # If the above fails, means back to the beginning (no previous data logged)
                # Set previous score to infinity
                post_param = ini_post_param.copy()
                distr = distr_map(post_param)
                prev_score = np.inf
            converged = False
        else:
            # Accept last step, try next one
            log_vi.add1(post_param, score=score_VI, KL=KL, mean=mean_val)
            v_new = -eta * (der_val + temperature * der_KL)

            v_new_flat = v_new.flatten()
            v_new_flat[index_no_train] = 0.0
            v_new = v_new_flat.reshape(distr_param_shape)
            v = v_new + v * momentum

            post_param = post_param + v

            prev_score = score_VI

            if (i % print_rec) == 0:
                blab(
                    silent,
                    f"Score at step {i}: {score_VI} (KL: {KL}, score:{mean_val})",
                )

            distr = distr_map(post_param)

            converged = np.max(np.abs(v)) < xtol

        i = i + 1

    opt_param, opt_score = log_vi.best()

    return OptimResultVI(
        opti_param=opt_param,
        converged=converged,
        opti_score=opt_score,
        hist_param=log_vi.distr_pars(),
        hist_score=log_vi.VI_scores(),
        end_param=post_param,
        log_vi=log_vi,
        bin_log_vi=bin_log_vi,
        sample_val=acc,
    )


def variational_inference(
    fun,
    distr_map: ProbaMap,
    prior_param: Optional[ProbaParam] = None,
    post_param: Optional[ProbaParam] = None,
    temperature: float = 1.0,
    prev_eval: Optional[Type[AccuSampleVal]] = None,
    index_train: Optional[list[int]] = None,
    VI_method: str = "corr_weights",
    eta: float = 0.05,
    chain_length: int = 10,
    per_step: int = 100,
    per_step_eval: int = 10000,
    xtol: float = 10**-8,
    k: Optional[int] = None,
    gen_decay: float = 0.0,
    momentum: float = 0.0,
    refuse_conf: float = 0.99,
    corr_eta: float = 0.5,
    parallel: bool = True,
    vectorized: bool = False,
    print_rec: int = 1,
    silent: bool = False,
    **kwargs,
) -> OptimResultVI:
    r"""Optimize Catoni's bound on a parametric set of distributions including the prior.

    Catoni's bound is defined as
        $$\int f(x) p(\theta, dx) + temp * KL(p(\theta), p0)$$

    The optimisation routine is based on Gradient descent with unbiased estimations of gradient.

    Args:
        fun: a scoring function
        distr_map: parametric set of distributions
        prior_param: parameter describing the prior distribution. Optional (if None, uses
            distr_map.ref_distr_param)
        post_param: parameter describing the initial guess of posterior distribution. Optional (if
            None, uses prior_param)
        temperature: the PAC-Bayesian temperature used to construct the posterior distribution
        prev_eval: AccuSampleVal object encoding previous evaluations of the function (useful for
            retraining). Optional
        index_train: indexes of the ProbaParam of the distr_map object which should be trained.
            Remaining parameters are left at their intial value.
        VI_method: which method is used for gradient descent. Either 'corr_weights' (default) or
            'knn' (not recommanded so far).
        eta: step size for gradient descent (normalised for the case where 'momentum'=0.0)
        chain_length: number of gradient steps.
        per_step: number of draws from the prior evaluated through 'fun' at each gradient step.
        per_step_eval: number of estimations of 'fun(x)' at each gradient step.
            For 'VI_method' = 'knn' only.
        xtol: termination criteria (somewhat unrealistic due to the probabilistic nature of
            gradient estimates.
        k: Number of sample evaluations used to estimate the gradient. Default is None
            (all evaluations are used).
        gen_decay: exponential decay schedule for the weights given to each generation for the
            gradient estimation. Default is 0.0 (equal contributions).
        momentum: momentum for the gradient descent. Speed at time t is
            s_t = momentum * s_{t-1} + (1 - momentum) * eta * Gradient_t
        refuse_conf: Step refusal mechanism. If VI_score_t - VI_score_{t-1} > 0 with confidence
            higher than 'refuse_conf', then revert to time t-1,  kill the accumulated speed
            (reset s to 0) and diminish step_size (eta = eta * corr_eta)
        corr_eta: contraction to step size when previous step increased the score.
        parallel: should 'fun' evaluations be parallelized? Default is True
        vectorized: is 'fun' vectorized? If True, 'parallel' is disregarded
        print_rec: How many gradient steps should there be prints
        silent: should there be any prints at all ? Default is False (there are prints)

    Further kwargs are passed to 'fun'.
    """
    if VI_method == "KNN":

        return optim_VI_knn(
            fun,
            distr_map=distr_map,
            prior_param=prior_param,
            post_param=post_param,
            temperature=temperature,
            prev_eval=prev_eval,
            index_train=index_train,
            eta=eta,
            chain_length=chain_length,
            per_step=per_step,
            per_step_eval=per_step_eval,
            xtol=xtol,
            k=k,
            momentum=momentum,
            # refuse_conf=refuse_conf, # Currently desactivated
            corr_eta=corr_eta,
            parallel=parallel,
            vectorized=vectorized,
            print_rec=print_rec,
            silent=silent,
            **kwargs,
        )

    if VI_method == "corr_weights":

        return optim_VI_wc(
            fun,
            distr_map=distr_map,
            temperature=temperature,
            prev_eval=prev_eval,
            prior_param=prior_param,
            post_param=post_param,
            index_train=index_train,
            eta=eta,
            chain_length=chain_length,
            per_step=per_step,
            xtol=xtol,
            k=k,
            gen_decay=gen_decay,
            momentum=momentum,
            refuse_conf=refuse_conf,
            corr_eta=corr_eta,
            parallel=parallel,
            vectorized=vectorized,
            print_rec=print_rec,
            silent=silent,
        )
