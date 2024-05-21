"""Finds ground state for antiferromagnetic Ising model.

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import numpy as np

import funsearch


@funsearch.run
def evaluate(el: tuple[int, ...]) -> int:
  """Returns the state with minimal energy."""
  offspring = energy(el)
  return min(offspring)


def energy(el: tuple[int, ...], N: int) -> float:
    """Compute the autocorrelation C(k) and the energy of a binary sequence.
    
    Args:
        sequence (np.ndarray): A 1-dimensional binary sequence.
        
    Returns:
        np.ndarray: The autocorrelation of the sequence for each lag.
        float: The energy of the autocorrelation.
    """
    N = len(el)
    mean = np.mean(el)
    var = np.var(el)
    autocorr = np.zeros(N)
    
    for k in range(N):
        if var == 0:
            autocorr[k] = 1 if k == 0 else 0  # Handle zero-variance case
            continue
        sum_corr = 0
        for i in range(N - k):
            sum_corr += (el[i] - mean) * (el[i + k] - mean)
        autocorr[k] = sum_corr / (var * (N - k))
    
    # Compute the energy as the sum of squares of autocorrelation values
    energy = np.sum(np.square(autocorr))
    
    return energy


@funsearch.evolve
def priority(el: tuple[int, ...], n: int) -> float:
  """Returns the energy of the new offspring.
  el is a tuple of length n with values 0-1.
  """
  return 0.0
