"""Obtains maximal independent sets.

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import itertools

import numpy as np

import funsearch


@funsearch.run
def evaluate(n: int) -> int:
  """Returns the size of an independent set."""
  independent_set = solve(n)
  return len(independent_set)


def solve(n: int) -> list[tuple[int, ...]]:
  """Gets independent set with maximal size.

  Args:
    num_nodes: The number of nodes of the base cyclic graph.
    n: The power we raise the graph to.

  Returns:
    A list of `n`-tuples in `{0, 1, 2, ..., num_nodes - 1}`.
  """
  num_nodes = 7
  to_block = np.array(list(itertools.product([-1, 0, 1], repeat=n)))

  # Powers in decreasing order for compatibility with `itertools.product`, so
  # that the relationship `i = children[i] @ powers` holds for all `i`.
  powers = num_nodes ** np.arange(n - 1, -1, -1)

  # Precompute the priority scores.
  children = np.array(
      list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
  scores = np.array([priority(tuple(child), n)
                     for child in children])

  # Build `max_set` greedily, using scores for prioritization.
  max_set = np.empty(shape=(0, n), dtype=np.int32)
  while np.any(scores != -np.inf):
    # Add a child with a maximum score to `max_set`, and set scores of
    # invalidated children to -inf, so that they never get selected.
    max_index = np.argmax(scores)
    child = children[None, max_index]  # [1, n]

    blocking = np.einsum(
        'cn,n->c', (to_block + child) % num_nodes, powers)  # [C]
    scores[blocking] = -np.inf
    max_set = np.concatenate([max_set, child], axis=0)

  return [tuple(map(int, el)) for el in max_set]


@funsearch.evolve
def priority(el: tuple[int, ...], n: int) -> float:
  """Returns the priority with which we want to add `el` to the set.

  Args:
    el: an n-tuple representing the element to consider whether to add.
    n: an integer, power of the graph.

  Returns:
    A number reflecting the priority with which we want to add `el` to the
    independent set.
  """
  return 0.