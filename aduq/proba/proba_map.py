r"""
Probability Map class.

For a parametric probability class, ProbaMap encodes the transform
    $$\alpha \rightarrow \mathbb{P}_\alpha.$$

This encoding also requires information on the derivative of the log density with respect to the
$\alpha$ parameter.
"""

import warnings
from typing import Callable, Iterable, List, Optional, Tuple, Type, Union

import numpy as np

from ..misc import blab, interpretation, par_eval, post_modif
from ._helper import _get_pre_shape, prod
from ._types import ProbaParam, SamplePoint, Samples
from .proba import Proba, tensorize
from .warnings import MissingShape, NegativeKL


class ProbaMap:  # pylint: disable=R0902
    r"""
    ProbaMap class.
    Mapping for parametric family of distribution

    Attributes:
        map, the actual mapping
        log_dens_der, a function of PriorParam outputing a function of PredictParam
        outputing a PriorParam such
        $\log_dens(x+ alpha, y ) - log_dens(x, y) \simeq log_dens_der(x)(y) . \alpha$
        for all small prior parameters alpha.
        distr_param_shape, the shape of the parameter describing the distribution
        sample_shape, the shared shape of the sample of every distribution in the family

    Note on log_dens_der input/output:
    Assuming that sqmple_shape = (s1, ..., sk) and that distr_param_shape is of shape (d1, ..., dp)
    Then log_dens_der(param) is a function which takes input of shape (n1, ..., nm, s1, ..., sk)
        and ouputs an array of shape (n1, ..., nm, d1, ..., dp)

    Future:
        Special case of integrate_der if the function is a Multivariate polynomial. Could be
        useful for gaussian distributions where the integration can be performed through exact
        computation.
    """

    def __init__(  # pylint: disable=R0913
        self,
        prob_map: Callable[[ProbaParam], Type[Proba]],
        log_dens_der: Callable[[ProbaParam], Callable[[Samples], np.ndarray]],
        ref_param: Optional[ProbaParam] = None,
        distr_param_shape: Optional[tuple] = None,
        sample_shape: Optional[tuple] = None,
    ):
        self.map = prob_map
        self.log_dens_der = log_dens_der

        if (ref_param is None) & (distr_param_shape is None):
            warnings.warn(
                "No shape information on expected distribution parameters",
                category=MissingShape,
            )
        if ref_param is not None:
            ref_param = np.array(ref_param)
        if (distr_param_shape is None) & (ref_param is not None):
            distr_param_shape = np.array(ref_param).shape
        if (distr_param_shape is not None) & (ref_param is None):
            # Try 0 parameter (can fail)
            warnings.warn(
                "\n".join(
                    [
                        "No reference parameter passed",
                        "Setting array full of 0 as reference parameter",
                    ]
                )
            )
            ref_param = np.zeros(distr_param_shape)

        self.distr_param_shape = distr_param_shape

        self.ref_param = ref_param

        if (sample_shape is None) & (ref_param is not None):
            sample_shape = self.map(ref_param).sample_shape

        if sample_shape is None:
            try:
                distr_param = np.zeros(distr_param_shape)
                sample_shape = self.map(distr_param).gen(1)[0].shape
            except Exception:  # pylint: disable=W0703
                warnings.warn("Could not infer shape of sample", MissingShape)
                sample_shape = None

        self.sample_shape = sample_shape

    def __call__(self, x):
        return self.map(x)

    def kl(  # pylint: disable=E0202
        self,
        param1: ProbaParam,
        param0: ProbaParam,
        n_sample: int = 1000,
    ) -> float:
        """Approximates the Kullback Leibler divergence between two distributions
        defined by their prior parameters.

        Args:
            param1, param0 are 2 prior parameters
            n_sample specifies how many points are used to estimate Kullback

        Output:
            kl(param1, param0) approximated as Sum_i(log(distrib1(phi_i)/distrib0(phi_i))
            with phi_i sampled through distrib1.gen (typically i.i.d.)

        Note:
            For a ProbaMap object obtained as the result of .reparametrize method with
            inherit_method=True, this method might be hidden behind a function attribute which
            is more efficient.
        """
        return self.map(param1).kl(self.map(param0), n_sample=n_sample)

    def grad_kl(  # pylint: disable=E0202
        self, param0: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the first distribution
        (nabla_{distrib1} kl(distrib1, distrib0))

        Args:
            param0, a distribution parameter

        Output:
            A closure taking as arguments:
                param1, a distribution parameter
                n_sample, an integer
            outputing a tuple with first element
                nabla_{param1}kl(param1, param0) approximated using a sample
                    phi_i of predictors generated through distrib1.gen (typically i.i.d.).
                kl(param1, param0) approximated using the same sample of predictors

        This method should be rewritten for families with closed form expressions of KL and
        KL gradient. The closure form is used to simplify caching computations related to param0
        (for instance inverse of covariance matrix for gaussian distributions).

        Note:
            For a ProbaMap object obtained as the result of .reparametrize method with
            inherit_method=True, this method might be hidden behind a function attribute which
            is more efficient.
        """

        distr0 = self.map(param0)
        log_dens0 = distr0.log_dens

        def fun(param1: ProbaParam, n_sample: int) -> tuple[ProbaParam, float]:
            distr1 = self.map(param0)
            pred_sample = distr1.gen(n_sample)

            log_dens1 = distr1.log_dens

            ev_log1 = log_dens1(pred_sample)
            ev_log0 = log_dens0(pred_sample)
            eval_log = np.array(ev_log1) - np.array(ev_log0)

            kl = np.mean(eval_log)
            eval_log = eval_log - kl  # shape n,

            log_der = self.log_dens_der(param1)

            eval_der = np.array(log_der(pred_sample))  # shape n, distr_param_shape
            kl_der = (
                np.tensordot(
                    eval_der, eval_log, (0, 0)  # (n, distr_param_shape)  # (n, )
                )
                / n_sample
            )  # (distr_param_shape)

            if kl < 0:
                warnings.warn(
                    f"Negative kl ({kl}) - consider raising n_sample parameter",
                    category=NegativeKL,
                )

            return kl_der, kl

        return fun

    def grad_right_kl(  # pylint: disable=E0202
        self, param1: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Compute the derivative of the Kullback--Leibler divergence with respect to the second
        distribution.

        """
        distr1 = self.map(param1)

        def fun(param0: ProbaParam, n_sample: int) -> tuple[ProbaParam, float]:

            # Derivative of KL evaluation
            sample = distr1(n_sample)
            der_kl = -np.array(self.log_dens_der(param0)(sample)).mean(0)

            # Evaluation of KL
            ev_log1 = distr1.log_dens(sample)
            ev_log0 = self(param0).log_dens(sample)
            eval_log = np.array(ev_log1) - np.array(ev_log0)

            kl = np.mean(eval_log)

            return der_kl, kl

        return fun

    def f_div(  # pylint: disable=E0202
        self,
        param1: ProbaParam,
        param0: ProbaParam,
        f: Callable[[float], float],
        n_sample: int = 1000,
    ) -> float:
        r"""Approximates the f-divergence between two distributions
        defined by their prior parameters.

        Args:
            param1, param0 are 2 prior parameters
            f, a convex function such that $f(1) = 0$ (No checks are performed).
            n_sample, number of points used to estimate the f-divergence

        Output:
            $D_f(distrib1, distrib0)$ approximated as $\sum_i(f(distrib1(\phi_i)/distrib0(\phi_i))$
            with $\phi_i$ sampled through distrib0.gen (typically i.i.d.)

        Note:
            For a ProbaMap object obtained as the result of .reparametrize method with
            inherit_method=True, this method might be hidden behind a function attribute which
            is more efficient.
        """
        return self.map(param1).f_div(self.map(param0), f=f, n_sample=n_sample)

    def grad_f_div(  # pylint: disable=E0202
        self,
        param0: ProbaParam,
        f: Callable[[float], float],
        f_der: Callable[[float], float],
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        r"""Approximates the gradient of the f-divergence between two distributions
        defined by their prior parameters, with respect to the first distribution.

        Args:
            param0, the parameter describing the second distribution.
            f, a convex function such that $f(1) = 0$ (No checks are performed).
                Should be vectorized
            f_der, the derivative of f. Should be vectorized
        """

        distr0 = self.map(param0)

        distr_param_shape = self.distr_param_shape
        n_dim = prod(distr_param_shape)

        def gradient_function(
            param1: ProbaParam, n_sample: int = 1000
        ) -> tuple[ProbaParam, float]:

            log_dens_der = self.log_dens_der(param1)
            distr1 = self.map(param1)

            def to_integrate(samples: Iterable[SamplePoint]) -> np.ndarray:

                out = np.zeros(
                    n_dim + 1
                )  # first n_dim for gradient computation, last dim kl

                delta_log_dens = np.array(distr1.log_dens(samples)) - np.array(
                    distr0.log_dens(samples)
                )
                ratio_dens = np.exp(delta_log_dens)  # shape n

                out[:-1] = np.tensordot(
                    log_dens_der(samples),  # shape (n, distr_param_shape)
                    f_der(ratio_dens),  # shape (n,)
                    (0, 0),
                ).flatten()
                out[-1] = f(ratio_dens) / ratio_dens
                return out

            result = distr1.integrate_vectorized(to_integrate, n_sample)
            return result[:-1].reshape(distr_param_shape), result[-1]

        return gradient_function

    def grad_right_f_div(  # pylint: disable=E0202
        self,
        param1: ProbaParam,
        f: Callable[[float], float],
        f_der: Callable[[float], float],
    ) -> Callable[[ProbaParam, int, bool], tuple[ProbaParam, float]]:
        r"""Approximates the gradient of the f-divergence between two distributions
        defined by their prior parameters, with respect to the second distribution.

        Args:
            param1, the parameter describing the first distribution.
            f, a convex function such that $f(1) = 0$ (No checks are performed).
            f_der, the derivative of f
        """
        return self.grad_f_div(
            param1, lambda x: x * f(1 / x), lambda x: f(1 / x) - (1 / x) * f_der(1 / x)
        )

    def read_distr(self, path, **kwargs) -> Proba:
        """Reads a distribution from a csv file"""
        param = np.genfromtxt(path, **kwargs)
        return self.map(param)

    def integrate_der(  # pylint: disable=R0913
        self,
        fun: Callable[[SamplePoint], Union[np.ndarray, float]],
        param: ProbaParam,
        n_sample: int = 1000,
        print_mean: bool = True,
        parallel: bool = False,
    ) -> Tuple[np.ndarray, Union[np.ndarray, float]]:
        r"""
        Compute the derivative of:
        F(alpha) -> \int f(x) exp(log_p(alpha, x)) dmu(x) / \int exp(log_p(alpha, x)) dmu(x)

        Computed through:
            $dF =
            \int f(x) d log_p(alpha, x)  dp(alpha, x)
            - \int d log_p(alpha, x)  dp(alpha, x) \int f(x) dp(alpha, x)$

        Note: For GD algorithms, integrate_der might not be the best option since constructing large
            samples/ evaluating the function on the sample might be prohibitive. Techniques are
            developped in the PAC-Bayes module to overcome these difficulties.
        """
        distr = self.map(param)
        log_der = self.log_dens_der(param)

        samples = distr(n_sample)

        evals = np.array(
            par_eval(fun, samples, parallel=parallel)
        )  # shape (n, out_shape)
        log_evals = np.array(log_der(samples))  # Shape (n, distr_param_shape)

        mean_eval, sd_eval = evals.mean(0), evals.std(0)

        blab((not print_mean), f"Mean of function: {mean_eval} (deviations: {sd_eval})")

        der = (
            np.tensordot(log_evals, evals - mean_eval, (0, 0)) / n_sample
        )  # shape (distr_param_shape, out_shape)

        # der = np.tensordot(log_evals, evals, (0, 0)) / n_sample

        return der, mean_eval

    def integrate_vectorized_der(
        self,
        fun: Callable[[Iterable[SamplePoint]], np.ndarray],
        param: ProbaParam,
        n_sample: int = 1000,
        silent: bool = True,
    ):
        r"""
        Compute the derivative of:
        F(alpha) -> \int f(x) exp(log_p(alpha, x)) dmu(x) / \int exp(log_p(alpha, x)) dmu(x)

        Computed through:
            $dF =
            \int f(x) d log_p(alpha, x)  dp(alpha, x)
            - \int d log_p(alpha, x)  dp(alpha, x) \int f(x) dp(alpha, x)$

        Note: For GD algorithms, integrate_der might not be the best option since constructing large
            samples/ evaluating the function on the sample might be prohibitive. Techniques are
            developped in the PAC-Bayes module to overcome these difficulties.

        Contrary to integrate_der, the function fun is assumed to be vectorized and output
        np.ndarray
        """

        distr = self.map(param)
        log_der = self.log_dens_der(param)

        samples = distr(n_sample)

        evals = fun(samples)  # Shape (n, out_shape)
        log_evals = np.array(log_der(samples))  # Shape (n, distr_param_shape)

        mean_eval, sd_eval = evals.mean(0), evals.std(0)

        blab(silent, f"Mean of function: {mean_eval} (deviations: {sd_eval})")

        der = (
            np.tensordot(log_evals, evals - mean_eval, (0, 0)) / n_sample
        )  # shape (distr_param_shape, out_shape)
        # der = np.tensordot(evals, log_evals, (0, 0)) / n_sample

        return der, mean_eval

    def reparametrize(  # pylint: disable=R0913
        self,
        transform: Callable[[ProbaParam], ProbaParam],
        der_transform: Callable[[ProbaParam], np.ndarray],
        inv_transform: Optional[Callable[[ProbaParam], ProbaParam]] = None,
        new_ref_param: Optional[ProbaParam] = None,
        distr_param_shape: Optional[Tuple[int, ...]] = None,
        inherit_methods: bool = True,
    ):
        """
        Given a transform B -> A, and assuming that the ProbaMap is parametrized by A,
        constructs a ProbaMap parametrized by B.
        Note: will fail if self.distr_param_shape attribute is None

        Arguments:
            transform, a transform between space B and A. Should be vectorized.
            der_transform, the Jacobian of the transform (der_transform(b) is a np.ndarray of shape
                (b1, ..., bn, a1, ..., am))
            inv_transform (optional), the inverse of transform, from B to A.
            distr_param_shape (optional), the shape of B.
            inherit_methods, whether the kl, grad_kl, grad_right_kl methods of the output uses the
                implementation from the initial ProbaMap object. Default is True.
        Output:
            A distribution map parametrized by B, with map self.map o transform

        Note on input output shape:
            Noting self.distr_param_shape = (a1, ..., am) and distr_param_shape = (b1, ..., bn),
            transform takes as input arrays of shape (n1, ..., nk, b1, ..., bm) and outputs array
            of shape (n1, ..., nk, a1, ..., an)

            der_transform takes as input arrays of shape (n1, ..., nk, b1, ..., bn) and outputs
            array of shape (n1, ..., nk, b1, ..., bn, a1, ..., am)

            inv_transform takes as input arrays of shape (n1, ..., nk, a1, ..., an) and outpus array
            of shape (n1, ..., nk, b1, ..., bn)

        Notably, if k = 0, then the outputs are of shape (a1, ..., an), (b1,..., bn, a1, ..., am) and
            (b1, ..., bn)  respectively

        Note: inv_transform is only used to assess the default reference parameter
            inherit_methods works by hiding the methods behind reimplemented functions. Not clean,
                but does the trick as far as tested.
        """

        if self.distr_param_shape is None:
            raise Exception(
                "distr_param_shape attribute must be specified to reparametrize"
            )

        def prob_map(param: ProbaParam) -> Proba:
            return self.map(transform(param))

        indices = list(np.arange(-len(self.distr_param_shape), 0))

        def log_dens_der(param: ProbaParam) -> Callable[[Samples], np.ndarray]:
            j_transform = der_transform(param)  # (new_shape,old_shape)
            old_param = transform(param)  # Shape (old_shape)
            old_log_der = self.log_dens_der(
                old_param
            )  # Will output (pre_shape, old_shape)

            def der(samples: Samples) -> np.ndarray:
                return np.tensordot(
                    old_log_der(samples),  # (pre_shape, old_shape)
                    j_transform,  # (new_shape, old_shape)
                    [indices, indices],
                )  # Shape (pre_shape, new_shape)

            return der

        if (new_ref_param is None) & (inv_transform is not None):
            if self.ref_param is not None:
                try:
                    new_ref_param = inv_transform(self.ref_param)
                except Exception:  # pylint: disable=W0703
                    warnings.warn("Could not infer the reference parameter")

        if new_ref_param is not None:
            if distr_param_shape is None:
                distr_param_shape = np.array(new_ref_param).shape
            else:
                new_param_shape = np.array(new_ref_param).shape
                check_shape = new_param_shape == distr_param_shape
                if not check_shape:
                    warnings.warn(
                        f""""
                        distr_param_shape indicated is not coherent with ref_param inferred.
                        Using {new_param_shape} as new shape
                        """
                    )
                    distr_param_shape = new_param_shape

        out = ProbaMap(
            prob_map,
            log_dens_der,
            ref_param=new_ref_param,
            distr_param_shape=distr_param_shape,
            sample_shape=self.sample_shape,
        )

        if inherit_methods:

            # Use old kl method
            def kl(
                param1: ProbaParam,
                param0: ProbaParam,
                n_sample: int = 1000,
            ):
                return self.kl(
                    transform(param1),
                    transform(param0),
                    n_sample=n_sample,
                )

            out.kl = kl  # Rewriting method by hiding it!

            # Use old grad kl method if possible
            def grad_kl(param0: ProbaParam):
                par0 = transform(param0)
                old_der = self.grad_kl(par0)

                def der(param1: ProbaParam, n_sample: int = 1000):

                    par1 = transform(param1)
                    j_transform = der_transform(param1)  # (new_shape, old_shape)
                    grad_kl, kl = old_der(par1, n_sample=n_sample)  # old_shape

                    return (
                        np.tensordot(
                            j_transform,  # (new_shape, old_shape)
                            grad_kl,  # (old_shape)
                            [indices, indices],  # (new_shape)
                        ),
                        kl,
                    )

                return der

            out.grad_kl = grad_kl  # Rewriting method by hiding it!

            def grad_right_kl(param1: ProbaParam):
                par1 = transform(param1)
                old_der = self.grad_right_kl(par1)

                def der(param0: ProbaParam, n_sample: int = 1000):
                    par0 = transform(param0)
                    j_transform = der_transform(param0)  #  (new_shape, old_shape)
                    grad_kl, kl = old_der(par0, n_sample=n_sample)  # old_shape

                    return (
                        np.tensordot(
                            j_transform,
                            grad_kl,
                            [indices, indices],
                        ),  # new_shape
                        kl,
                    )

                return der

            out.grad_right_kl = grad_right_kl  # Rewriting method by hiding it!

        return out

    def subset(
        self,
        sub_indexes: list[int],
        default_param: Optional[ProbaParam] = None,
        inherit_methods: bool = True,
    ):
        r"""
        Define a new ProbaMap object from partial ProbaParam object.

        For a distribution map $M:(\theta_1, \dots, \theta_n)-> \mathcal{P}_\theta$,
        output the distribution map
            $$(\theta_{id_1}, \dots)$ -> M((\theta_1^*, \dots, \theta_{id_1}, \dots \theta_n^*))$
        where $\theta_i^*$ are fixed values inferred from default param.

        Exemple: construct a Gaussian map with fixed mean from the standard gaussian map.
        While could this also be achieved through reparametrize function, we can avoid using the
        sparse der_transform matrix

        Arguments:
            sub_indexes: list of indexes settable in the resulting ProbaMap object
            default_param: default parameter values (used for the non settable parameters). Default
                is None, amounts to self.ref_param .
            inherit_methods, whether the kl, grad_kl, grad_right_kl methods of the output uses the
                implementation from the initial ProbaMap object. Default is True.
        Output:
            The distribution map taking a reduced ProbaMap as input.
        """
        if default_param is None:
            default_param = self.ref_param.copy()

        default_param = np.array(default_param).flatten()

        def to_param(par: ProbaParam) -> ProbaParam:
            full_par = default_param.copy()  # No side effect on default_param
            full_par[sub_indexes] = par
            return full_par.reshape(self.distr_param_shape)

        def project(pars: Iterable[ProbaParam]) -> Iterable[ProbaParam]:
            pars = np.array(pars)
            pre_shape = pars.shape[: len(self.distr_param_shape)]

            if pre_shape == ():
                return pars.flatten()[sub_indexes]

            n_distr_param = prod(self.distr_param_shape)
            return pars.reshape(prod(pre_shape), n_distr_param)[:, sub_indexes].reshape(
                pre_shape + (n_distr_param,)
            )

        # Making use of composition here to define new prob_map and log_dens_der
        set_up_param = interpretation(to_param)
        prob_map = set_up_param(self.map)
        super_set_up_mod = post_modif(post_modif(project))
        log_dens_der = super_set_up_mod(set_up_param(self.log_dens_der))

        # Convert reference param
        ref_param = project(default_param)
        out = ProbaMap(
            prob_map,
            log_dens_der,
            ref_param=ref_param,
            distr_param_shape=ref_param.shape,
            sample_shape=self.sample_shape,
        )

        if inherit_methods:
            # Use old kl method
            def kl(
                param1: ProbaParam,
                param0: ProbaParam,
                n_sample: int = 1000,
            ):
                return self.kl(
                    to_param(param1),
                    to_param(param0),
                    n_sample=n_sample,
                )

            out.kl = kl

            # Use old grad kl method if possible
            def grad_kl(param0: ProbaParam):
                par0 = to_param(param0)
                old_der = self.grad_kl(par0)

                def der(param1: ProbaParam, n_sample: int = 1000):
                    par1 = to_param(param1)
                    grad_kl, kl = old_der(par1, n_sample=n_sample)
                    return (project(grad_kl), kl)

                return der

            out.grad_kl = grad_kl

            def grad_right_kl(param1: ProbaParam):
                par1 = to_param(param1)
                old_der = self.grad_right_kl(par1)

                def der(param0: ProbaParam, n_sample: int = 1000):
                    par0 = to_param(param0)
                    grad_kl, kl = old_der(par0, n_sample=n_sample)

                    return (project(grad_kl), kl)

                return der

            out.grad_right_kl = grad_right_kl

        return out

    def transform(
        self,
        transform: Callable[[Samples], Samples],
        inv_transform: Callable[[Samples], Samples],
        der_transform: Optional[Callable[[Samples], np.ndarray]] = None,
    ):
        r"""
        Transform the Class of probability $X_\theta \sim \mathbb{P}_{\theta}$ to the class of probability
            $transform(X_\theta)$

        Important:
            transform MUST be bijective, else computations for log_dens_der, kl, grad_kl, grad_right_kl will fail.


        CAUTION:
            Everything possible is done to insure that the reference distribution remains
            Lebesgue IF the original reference distribution is Lebesgue. This requires access
            to the derivative of the transform (more precisely its determinant). If this can
            not be computed, then the log_density attribute will no longer be with reference to
            the standard Lebesgue measure.

            Still,
                distr1.transform(f, inv_f).log_dens(x) - distr2.transform(f, inv_f).log_dens(x)
            acccurately describes
                log (d distr1 / d distr2 (x)).

            If only ratios of density are to be investigated, der_transform can be disregarded.

            Due to this, log_dens_der, kl, grad_kl, grad_right_kl will perform satisfactorily.

        Dimension formating:
            If distr outputs samples of shape (s1, ..., sk), and transforms maps them to (t1, ..., tl)
             then the derivative should be shaped
                (s1, ..., sk, t1, ..., tl).
            The new distribution will output samples of shape (t1, ..., tl).

            Moreover, transform, inv_transform and der_transform are assumed to be vectorized, i.e.
            inputs of shape (n1, ..., np, s1, ..., sk) will result in outputs of shape
                (n1, ..., np, t1, ..., tl ) for transform, (n1, ..., np, s1, ..., sk, t1, ..., tl ) for
                der_transform

        Output formating:
            The format of the output depends on the behavior of transform. If transform outputs np.ndarray,
            then the resulting distribution will have np_out==True.

        Future:
            Decide whether der_transform must output np.ndarray or not. Currently, does not have to (but could be less
            efficient since reforce array creation.)
        """

        def new_map(x: ProbaParam) -> Proba:
            return self.map(x).transform(transform, inv_transform, der_transform)

        def new_log_dens_der(
            x: ProbaParam,
        ) -> Callable[[Samples], np.ndarray]:
            log_dens_der_fun = self.log_dens_der(x)

            def new_func(samples: Samples) -> np.ndarray:
                return log_dens_der_fun(inv_transform(samples))

            return new_func

        out = ProbaMap(
            prob_map=new_map,
            log_dens_der=new_log_dens_der,
            ref_param=self.ref_param,
            distr_param_shape=self.distr_param_shape,
            sample_shape=self.map(self.ref_param).sample_shape,
        )

        # kl computations are performed using old map
        out.kl = self.kl
        out.grad_kl = self.grad_kl
        out.grad_right_kl = self.grad_right_kl
        out.f_div = self.f_div
        out.grad_f_div = self.grad_f_div
        out.grad_right_f_div = self.grad_right_f_div

        return out

    def forget(self):
        """
        Returns a ProbaMap object with standard implementation for methods.
        """
        return ProbaMap(
            prob_map=self.map,
            log_dens_der=self.log_dens_der,
            ref_param=self.ref_param,
            distr_param_shape=self.distr_param_shape,
            sample_shape=self.sample_shape,
        )


def map_tensorize(*maps: ProbaMap) -> ProbaMap:  # pylint: disable=R0914
    """
    Given a collection of distribution maps

        mu1 -> Proba1, mu2 -> Proba2, ...

    returns the map (mu1, mu2 ... ) -> Proba1 x Proba2 x ...

    For the time being, we consider only distributions on 1D arrays.

    """
    distr_param_shapes = [prob_map.distr_param_shape for prob_map in maps]
    distr_param_len = [
        prod(distr_param_shape) for distr_param_shape in distr_param_shapes
    ]
    distr_param_shape_tot = (sum(distr_param_len),)

    sample_shapes = [prob_map.sample_shape for prob_map in maps]
    sample_len = [prod(sample_shape) for sample_shape in sample_shapes]
    sample_shape_tot = (sum(sample_len),)

    pivot_param = np.cumsum([0] + distr_param_len)
    pivot_sample = np.cumsum([0] + sample_len)

    def split_param(param: ProbaParam) -> List[ProbaParam]:
        param = np.array(param)
        return [
            param[p0:p1].reshape(distr_param_shape)
            for p0, p1, distr_param_shape in zip(
                pivot_param[:-1], pivot_param[1:], distr_param_shapes
            )
        ]

    def _decomp_sample(samples: Samples) -> List[Samples]:
        pre_shape = _get_pre_shape(samples, sample_shape_tot)

        samples = samples.reshape((prod(pre_shape), samples.shape[-1]))

        return [
            samples[:, p0:p1].reshape(pre_shape + sample_shape)
            for p0, p1, sample_shape in zip(
                pivot_sample[:-1], pivot_sample[1:], sample_shapes
            )
        ]

    def prob_map(param: ProbaParam) -> Proba:
        params = split_param(param)
        return tensorize(
            *tuple(p_map(par) for par, p_map in zip(params, maps)), flatten=True
        )

    def log_dens_der(param: ProbaParam) -> Callable[[Samples], np.ndarray]:
        params = split_param(param)
        ldds = [p_map.log_dens_der(par) for par, p_map in zip(params, maps)]

        def ldd(samples: Samples) -> np.ndarray:
            samples = np.array(samples)
            pre_shape = samples.shape[:-1]
            n_samples = prod(pre_shape)
            samples = samples.reshape((n_samples, samples.shape[-1]))

            samples = _decomp_sample(
                samples
            )  # list[ Array of shape (n_samples, samp_shape_i)]

            ders = [ldd_i(sample) for sample, ldd_i in zip(samples, ldds)]

            der = np.zeros((n_samples,) + distr_param_shape_tot)
            for p0, p1, der_i in zip(pivot_param[:-1], pivot_param[1:], ders):
                der[:, p0:p1] = der_i.reshape((n_samples, prod(der_i.shape[1:])))

            return der.reshape(pre_shape + distr_param_shape_tot)

        return ldd

    ref_params = [p_map.ref_param for p_map in maps]
    if not np.any([ref_params is None for ref_param in ref_params]):
        ref_param = np.zeros(distr_param_shape_tot)
        for p0, p1, ref in zip(pivot_param[:-1], pivot_param[1:], ref_params):
            ref_param[p0:p1] = ref.flatten()
    else:
        ref_param = None

    distr_map = ProbaMap(
        prob_map=prob_map,
        log_dens_der=log_dens_der,
        ref_param=ref_param,
        distr_param_shape=distr_param_shape_tot,
        sample_shape=sample_shape_tot,
    )

    # Overwrite kl method
    def kl(
        param1: ProbaParam, param0: ProbaParam, n_sample: int = 1000
    ):  # pylint: disable=W0613
        param1_s = split_param(param1)
        param0_s = split_param(param0)

        return np.sum(
            [
                p_map.kl(param1_i, param0_i, n_sample)
                for p_map, param1_i, param0_i in zip(maps, param1_s, param0_s)
            ]
        )

    distr_map.kl = kl

    def grad_kl(
        param0: ProbaParam,
    ) -> Callable[[ProbaParam, int], Tuple[ProbaParam, float]]:
        param0_s = split_param(param0)
        grad_kls = [d_map.grad_kl(param0_i) for d_map, param0_i in zip(maps, param0_s)]

        def der(param1: ProbaParam, n_sample: int = 1000):
            param1_s = split_param(param1)

            info = [
                grad_kl_i(param1_i, n_sample)
                for grad_kl_i, param1_i in zip(grad_kls, param1_s)
            ]

            ders = [x[0].flatten() for x in info]
            kls = [x[1] for x in info]

            der = np.zeros(distr_param_shape_tot)
            for p0, p1, der_i in zip(pivot_param[:-1], pivot_param[1:], ders):
                der[p0:p1] = der_i

            return (der, np.sum(kls))

        return der

    distr_map.grad_kl = grad_kl

    def grad_right_kl(
        param1: ProbaParam,
    ) -> Callable[[ProbaParam, int], Tuple[ProbaParam, float]]:
        param1_s = split_param(param1)
        grad_kls = [d_map.grad_kl(param1_i) for d_map, param1_i in zip(maps, param1_s)]

        def der(param0: ProbaParam, n_sample: int = 1000):
            param0_s = split_param(param0)

            info = [
                grad_kl_i(param0_i, n_sample)
                for grad_kl_i, param0_i in zip(grad_kls, param0_s)
            ]

            ders = [x[0].flatten() for x in info]
            kls = [x[1] for x in info]

            der = np.zeros(distr_param_shape_tot)
            for p0, p1, der_i in zip(pivot_param[:-1], pivot_param[1:], ders):
                der[p0:p1] = der_i

            return (der, np.sum(kls))

        return der

    distr_map.grad_right_kl = grad_right_kl

    return distr_map
