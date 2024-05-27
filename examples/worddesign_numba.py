import numpy as np
from numba import njit, prange
from typing import List, Tuple

@njit
def generate_initial_sequences(n: int) -> np.ndarray:
    """Generates all sequences of length n using nucleotides {A, C, G, T}."""
    num_sequences = 4 ** n
    all_sequences = np.empty((num_sequences, n), dtype=np.int32)
    for i in range(num_sequences):
        for j in range(n):
            all_sequences[i, j] = (i // (4 ** (n - j - 1))) % 4
    return all_sequences

@njit
def is_complementary(seq1, seq2):
    """ Vectorized complement check with direct complement mapping. """
    mapping = np.array([3, 2, 1, 0], dtype=np.int32)  # T, G, C, A
    seq1_reversed = seq1[::-1]
    seq2_complement = mapping[seq2]
    mismatches = np.sum(seq1_reversed != seq2_complement)
    return mismatches >= 4

def priority(el: Tuple[int, ...], n: int) -> float:
    """Returns the priority with which we want to add `element` to the word set."""
    return 0.0

@njit
def check_conditions_post(wordset: np.ndarray) -> np.ndarray:
    """Applies post-collection checks to ensure 3 constraints are met: Hamming distance,
    Watson-Crick complementarity, and specific GC content.
    """
    n = wordset.shape[0]
    mask_diffs = np.ones(n, dtype=np.bool_)
    mask_complementary = np.ones(n, dtype=np.bool_)
    mask_gc_content = np.array([np.sum((word == 1) | (word == 2)) == 4 for word in wordset])

    for i in prange(n):
        for j in prange(i + 1, n):
            if np.sum(wordset[i] != wordset[j]) < 4:
                mask_diffs[i] = mask_diffs[j] = False

    for i in prange(n):
        for j in prange(i, n):
            if not is_complementary(wordset[i], wordset[j]):
                mask_complementary[i] = mask_complementary[j] = False

    final_mask = mask_diffs & mask_complementary & mask_gc_content
    return wordset[final_mask]

@njit
def solve_numba(n: int, priorities: np.ndarray, all_vectors: np.ndarray) -> List[np.ndarray]:
    wordset = []
    while np.any(priorities != -np.inf):
        max_index = np.argmax(priorities)
        selected_vector = all_vectors[max_index]

        # Vectorized calculation of differences using broadcasting
        differences = np.sum(all_vectors != selected_vector, axis=1)
        mask_complementary = np.array([is_complementary(selected_vector, v) for v in all_vectors])

        # Invalidate vectors that are too similar, not sufficiently complementary, or do not meet GC content
        gc_content = np.sum((all_vectors == 1) | (all_vectors == 2), axis=1)
        mask_gc_content = gc_content == n / 2
        mask_invalid = (differences < 4) | (~mask_complementary) | (~mask_gc_content)
        priorities[mask_invalid] = -np.inf
        priorities[max_index] = -np.inf

        wordset.append(selected_vector)

    return wordset

def solve(n: int) -> np.ndarray:
    all_vectors = generate_initial_sequences(n)
    priorities = np.zeros(len(all_vectors), dtype=np.float32)  # All priorities are initially zero

    wordset = solve_numba(n, priorities, all_vectors)
    
    # Post-collection filtering
    wordset = np.array(wordset)
    wordset = check_conditions_post(wordset)
    return wordset

def evaluate(n: int) -> int:
    """Returns the size of an n-dimensional set of DNA sequences."""
    wordset = solve(n)
    return len(wordset)
