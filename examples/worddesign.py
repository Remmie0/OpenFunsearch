"""Finds large sets of DNA words.

Return code for Python method priority_vX, where every iteration of priority_vX improves on previous iterations.

Be creative, try to not make the changes to big.  
Find an algorithm that improves over the other priority functions.

only return code without explanation or something else, this is to reduce overhead and to use output in a sequence. Only return 1 function in total.

Return the code in full function such as:

def priority(el: Tuple[int, ...], n: int) -> float:

    'Function'

    return

"""
import itertools
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import funsearch


@funsearch.run
def evaluate(n: int) -> int:
  """Returns the size of an n-dimensional set of DNA sequences."""
  wordset = solve(n)
  return len(wordset)

def generate_initial_sequences(n: int) -> np.ndarray:
  """Generates all sequences of length n using nucleotides {A, C, G, T}."""
  bases = [0, 1, 2, 3]  # A, C, G, T
  all_sequences = [list(seq) for seq in itertools.product(bases, repeat=n)]
  return np.array(all_sequences, dtype=np.int32)

def solve(n: int) -> np.ndarray:
  all_vectors = generate_initial_sequences(n)
  priorities = np.array([priority(tuple(vector), n) for vector in all_vectors])

  wordset = []
  while np.any(priorities != -np.inf):
    max_index = np.argmax(priorities)
    selected_vector = all_vectors[max_index]

    # Vectorized calculation of differences using broadcasting
    differences = np.sum(all_vectors != selected_vector, axis=1)
    mask_complementary = np.array([is_complementary(selected_vector, v) for v in all_vectors])

    # Invalidate vectors that are too similar, not sufficiently complementary, or do not meet GC content
    mask_gc_content = np.sum((all_vectors == 1) | (all_vectors == 2), axis=1) == n/2
    mask_invalid = (differences < 4) | (~mask_complementary) | (~mask_gc_content)
    priorities[mask_invalid] = -np.inf
    priorities[max_index] = -np.inf

    wordset.append(selected_vector)

  # Post-collection filtering
  wordset = np.array(wordset)
  wordset = check_conditions_post(wordset)
  return wordset

def is_complementary(seq1, seq2):
  """ Vectorized complement check with direct complement mapping. """
  mapping = np.array([3, 2, 1, 0])  
  seq1_reversed = np.flip(seq1)
  seq2_complement = mapping[seq2]
  mismatches = np.sum(seq1_reversed != seq2_complement)
  return mismatches >= 4

def check_conditions_post(wordset: np.ndarray) -> np.ndarray:
  """Applies post-collection checks to ensure 3 constraints are met: Hamming distance,
  Watson-Crick complementarity, and specific GC content.
  """
  n = len(wordset)
  mask_diffs = np.ones(n, dtype=bool)
  mask_complementary = np.ones(n, dtype=bool)
  mask_gc_content = np.array([(np.sum((word == 1) | (word == 2))) == 4 for word in wordset])

  for i in range(n):
    for j in range(i + 1, n):
      if np.sum(wordset[i] != wordset[j]) < 4:  
        mask_diffs[i] = mask_diffs[j] = False
    
  for i in range(n):
    for j in range(i, n):
      if not is_complementary(wordset[i], wordset[j]):
        mask_complementary[i] = mask_complementary[j] = False

  final_mask = mask_diffs & mask_complementary & mask_gc_content
  return wordset[final_mask]

@funsearch.evolve
def priority(el: Tuple[int, ...], n: int) -> float:
  """
  Returns the priority with which we want to add `element` to the word set.
  el is a tuple of length n, with values 0-3 corresponding to A, C, G, T.
  """

  return 0.