import numpy as np
import matplotlib.pyplot as plt
import time

def set_seed(seed_value=42):
    """Sets the random seed for NumPy's random number generator."""
    np.random.seed(seed_value)

def binary_to_hex(x):
    """Converts a binary numpy array to a hexadecimal string."""
    binary_string = ''.join(str(int(bit)) for bit in x)
    hexadecimal = f"{int(binary_string, 2):X}"
    return hexadecimal

def generate_symmetric_matrix(n):
    """Generates a symmetric matrix with mixed positive and negative values."""
    A = np.random.randn(n, n)  # Generate a matrix with values from a normal distribution
    A_symmetric = (A + A.T) / 2  # Make the matrix symmetric
    return A_symmetric

def evaluate(instances: dict, N):
    """Evaluate heuristic function on a set of BQP instances."""
    objective_values = []
    start_time = time.time()
    for name, instance in instances.items():
        Q = instance['Q']
        c = instance['c']
        n = Q.shape[0]
        x = np.random.randint(2, size=n)  # Initial random binary vector
        optimized_x, min_value = optimize_bqp(Q, c, x, N)
        objective_values.append(min_value)
    end_time = time.time()
    time_taken = end_time - start_time
    return -np.mean(objective_values), time_taken

def optimize_bqp(Q, c, x_initial, N):
    """Optimizes a binary vector x for the given BQP using a heuristic."""
    current_x = x_initial
    current_value = objective_function(Q, c, current_x)
    for _ in range(N):
        neighbors = generate_neighbors(current_x)
        neighbor_values = [objective_function(Q, c, neighbor) for neighbor in neighbors]
        best_index = np.argmin(neighbor_values)
        if neighbor_values[best_index] < current_value:
            current_x = neighbors[best_index]
            current_value = neighbor_values[best_index]
    return current_x, current_value

def objective_function(Q, c, x):
    """Calculates the objective function value for a binary vector x."""
    return np.dot(x, Q.dot(x)) + np.dot(c, x)

def generate_neighbors(x):
    """Generates neighboring solutions by flipping each bit of x."""
    neighbors = []
    for i in range(len(x)):
        neighbor = np.copy(x)
        neighbor[i] = 1 - neighbor[i]
        neighbors.append(neighbor)
    return neighbors

# Set the random seed for reproducibility
set_seed(42)

# Parameters
n = 1000  # dimension of the matrix and vector
Q = generate_symmetric_matrix(n)
c = np.random.randn(n)  # random vector with negative and positive values
instances = {'instance': {'Q': Q, 'c': c}}

# Range of N values to evaluate
N_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
scores = []
times = []

# Evaluate for different N values
for N in N_values:
    score, time_taken = evaluate(instances, N)
    scores.append(score)
    times.append(time_taken)
    print(f"N={N}, Score={score:.2f}, Time={time_taken:.2f}s")

best_score = max(scores)
errors = [best_score - score + 1 for score in scores]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].loglog(N_values, times, marker='o')
axs[0].set_title('Execution Time vs. N')
axs[0].set_xlabel('N (depth of evaluation)')
axs[0].set_ylabel('Time (seconds)')

axs[1].loglog(N_values, errors, marker='o', color='r')
axs[1].set_title('Error vs. N')
axs[1].set_xlabel('N (depth of evaluation)')
axs[1].set_ylabel('Error (log scale)')

plt.tight_layout()
plt.show()

