"""
Functions for testing whether subclasses of Proba and ProbaMap are well implemented.

Useful to check that:
- rewritten kl/grad_kl/grad_right_kl give coherent results
- log_dens and log_dens_der are coherent
"""
import numpy as np

from .proba_map import ProbaMap


def check_grad_kl(
    pmap: ProbaMap,
    n1: int = 10,
    n2: int = 10,
    n3: int = 10,
    delta: float = 10**-7,
    tol: float = 10**-2,
):
    """
    Check coherence of grad_kl in a ProbaMap.
    """
    pshape = pmap.distr_param_shape
    accu = np.zeros((n1, n2, n3))
    for i in range(n1):
        par1 = np.random.normal(0, 1, pshape)
        grad_kl_fun = pmap.grad_kl(par1)
        for j in range(n2):
            par2 = np.random.normal(0, 1, pshape)
            grad_kl, kl = grad_kl_fun(par2)
            for k in range(n3):
                par3 = np.random.normal(0, delta, pshape)
                accu[i, j, k] = (pmap.kl(par2 + par3, par1) - kl) / np.sum(
                    grad_kl * par3
                )
    max_err = np.max(np.abs(accu - 1))
    if max_err > tol:
        raise Exception(
            f"The implementation of grad_kl for {pmap} seems false (max error for {n1}*{n2}*{n3} repeats of {max_err})"
        )
    print(f"Max error for {n1}*{n2}*{n3} repeats of {max_err}")


def check_grad_right_kl(
    pmap: ProbaMap,
    n1: int = 10,
    n2: int = 10,
    n3: int = 10,
    delta: float = 10**-7,
    tol: float = 10**-2,
):
    """
    Check coherence of grad_right_kl in a ProbaMap.
    """
    pshape = pmap.distr_param_shape
    accu = np.zeros((n1, n2, n3))
    for i in range(n1):
        par1 = np.random.normal(0, 1, pshape)
        grad_right_kl_fun = pmap.grad_right_kl(par1)
        for j in range(n2):
            par2 = np.random.normal(0, 1, pshape)
            grad_right_kl, kl = grad_right_kl_fun(par2)
            for k in range(n3):
                par3 = np.random.normal(0, delta, pshape)
                accu[i, j, k] = (pmap.kl(par1, par2 + par3) - kl) / np.sum(
                    grad_right_kl * par3
                )
    max_err = np.max(np.abs(accu - 1))
    if max_err > tol:
        raise Exception(
            f"The implementation of grad_right_kl for {pmap} seems false (max error for {n1}*{n2}*{n3} repeats of {max_err}"
        )
    print(f"Max error for {n1}*{n2}*{n3} repeats of {max_err}")


def check_log_dens_der(
    pmap: ProbaMap,
    n1: int = 10,
    n2: int = 10,
    n3: int = 10,
    delta: float = 10**-7,
    tol: float = 10**-2,
):
    """
    Check coherence of log_dens_der in a ProbaMap.
    """
    accu = np.zeros((n1, n2, n3))

    prob_param_shape = pmap.distr_param_shape

    for i in range(n1):
        prob_param = np.random.normal(0, 1, prob_param_shape)

        log_dens_der_fun = pmap.log_dens_der(prob_param)
        prob = pmap(prob_param)
        log_dens_fun = prob.log_dens

        for j in range(n2):
            delta_par = np.random.normal(0, delta, prob_param_shape)
            prob_param2 = prob_param + delta_par

            prob2 = pmap(prob_param2)
            log_dens_fun2 = prob2.log_dens

            samples = prob(n3)

            for k, sample in enumerate(samples):
                accu[i, j, k] = (log_dens_fun2(sample) - log_dens_fun(sample)) / np.sum(
                    delta_par * log_dens_der_fun(sample)
                )

    max_err = np.max(np.abs(accu - 1))
    if max_err > tol:
        raise Exception(
            f"The implementation of log_dens_der for {pmap} seems false (max error for {n1}*{n2}*{n3} repeats of {max_err}"
        )
    print(f"Max error for {n1}*{n2}*{n3} repeats of {max_err}")
