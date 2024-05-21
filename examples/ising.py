"""Finds ground state Ising model.

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import itertools
import numpy as np
# import funsearch


# @funsearch.run
def evaluate(n: int) -> int:
    """Returns the minimal energy of the ground states."""
    groundstates = solve(n)
    return min(np.sum(groundstates, axis=1))


def should_invalidate(current_vector, test_vector):
    """Define your invalidation logic here."""
    return np.sum(current_vector != test_vector) == 0


def solve(n: int) -> np.ndarray:
    """Returns ground states in `n` dimensions."""
    all_vectors = np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.int32)
    priorities = np.array([priority(tuple(vector), n) for vector in all_vectors])
    
    groundstate = np.empty(shape=(0, n), dtype=np.int32)
    blocked_vectors = set()

    while np.any(priorities != -np.inf):
        max_index = np.argmax(priorities)
        vector = all_vectors[max_index]
        
        # Add vector and its permutations to the blocked set
        for perm in itertools.permutations(vector):
            blocked_vectors.add(tuple(perm))

        priorities[max_index] = -np.inf  # Invalidate this vector
        groundstate = np.concatenate([groundstate, [vector]], axis=0)

        # Invalidate all permutations of current vector in priorities
        for i, vec in enumerate(all_vectors):
            if tuple(vec) in blocked_vectors:
                priorities[i] = -np.inf

    print(f"groundstate = {groundstate}")
    return groundstate


# @funsearch.evolve
def priority(el: tuple[int, ...], n: int) -> float:
    """Returns the priority with which we want to add `element` to the cap set.
    el is a tuple of length n with values 0-2.
    """
    return 0.0

evaluate(3)