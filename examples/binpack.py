"""Finds heuristics for online 1d binpacking.

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import numpy as np

import funsearch

@funsearch.run
def evaluate(instances: dict) -> float:
  """Evaluate heuristic function on a set of online binpacking instances."""
  # List storing number of bins used for each instance.
  num_bins = []
  # Perform online binpacking for each instance.
  for name in instances:
    instance = instances[name]
    capacity = instance['capacity']
    items = instance['items']
    # Create num_items bins so there will always be space for all items,
    # regardless of packing order. Array has shape (num_items,).
    bins = np.array([capacity for _ in range(instance['num_items'])])
    # Pack items into bins and return remaining capacity in bins_packed, which
    # has shape (num_items,).
    _, bins_packed = online_binpack(items, bins)
    # If remaining capacity in a bin is equal to initial capacity, then it is
    # unused. Count number of used bins.
    num_bins.append((bins_packed != capacity).sum())
  # Score of heuristic function is negative of average number of bins used
  # across instances (as we want to minimize number of bins).
  return -np.mean(num_bins)

def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
  """Returns indices of bins in which item can fit."""
  return np.nonzero((bins - item) >= 0)[0]

def online_binpack(items: tuple[float, ...], bins: np.ndarray) -> tuple[list[list[float]], np.ndarray]:
  """Performs online binpacking of `items` into `bins`."""
  # Track which items are added to each bin.
  packing = [[] for _ in bins]
  # Add items to bins.
  for item in items:
    # Extract bins that have sufficient space to fit item.
    valid_bin_indices = get_valid_bin_indices(item, bins)
    # Score each bin based on heuristic.
    priorities = priority(item, bins[valid_bin_indices])
    # Add item to bin with highest priority.
    best_bin = valid_bin_indices[np.argmax(priorities)]
    bins[best_bin] -= item
    packing[best_bin].append(item)
  # Remove unused bins from packing.
  packing = [bin_items for bin_items in packing if bin_items]
  return packing, bins


@funsearch.evolve
def priority(item: float, bins: np.ndarray) -> np.ndarray:
  """Returns priority with which we want to add item to each bin.

  Args:
    item: Size of item to be added to the bin.
    bins: Array of capacities for each bin.

  Return:
    Array of same size as bins with priority score of each bin.
  """
  return -(bins - item)