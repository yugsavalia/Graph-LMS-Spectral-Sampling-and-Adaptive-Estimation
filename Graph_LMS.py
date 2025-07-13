import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh

def greedy_sampling(B, K):
    """
    Greedily select K nodes to maximize the smallest eigenvalue of B D B.
    """
    N = B.shape[0]
    S = []  # selected set
    remaining = set(range(N))
    for _ in range(K):
        best_node, best_min_ev = None, -np.inf
        for i in remaining:
            S_trial = S + [i]
            D_trial = np.diag([1 if j in S_trial else 0 for j in range(N)])
            BDB = B @ D_trial @ B 
            # Only consider non-zero frequencies, so we restrict to K 
            # by looking at the top K eigenvalues of BDB
            evs = eigvalsh(BDB)
            min_ev = min(evs[-len(S_trial):])  # smallest among top S_trial
            if min_ev > best_min_ev:
                best_min_ev = min_ev
                best_node = i
        S.append(best_node)
        remaining.remove(best_node)
    return S

def run_lms_algorithm(A, nodes, F=None, mu=0.2, max_iters=1000, tolerance=1e-2):
    """
    Parameters:
    - A: Adjacency matrix
    - nodes: Number of nodes
    - F: Frequency indices (default: [0, 1])
    - mu: Step size
    - max_iters: Maximum iterations
    - tolerance: Convergence tolerance
    """
    
    if F is None:
        F = [0, 1]
    
    # Calculate graph matrices
    Deg = np.diag(A.sum(axis=1))
    L = Deg - A
    
    eigvals, U = np.linalg.eigh(L)
    
    # B (projection matrix for selected frequencies)
    Sigma_F = np.diag([1 if i in F else 0 for i in range(len(eigvals))])
    B = U @ Sigma_F @ U.T
    
    # Greedy sampling 
    K = len(F)
    S_opt = greedy_sampling(B, K)
    D = np.diag([1 if i in S_opt else 0 for i in range(nodes)])
    
    # Given signal
    x0_unfiltered = np.random.randn(nodes)
    x0 = B @ x0_unfiltered
    
    # LMS initialization
    x = B @ np.zeros(nodes)  # random starting point 
    
    # Collect estimates over iterations
    records = []
    
    for n in range(max_iters + 1):
        error_norm = np.linalg.norm(x - x0)
        
        # Record the current estimate
        row = {'Iteration': n}
        for i in range(len(x)):
            row[f'x[{i}]'] = round(x[i], 4)
        records.append(row)
        
        # Stop if error is within tolerance
        if error_norm < tolerance:
            print(f"Converged at iteration {n} with error {error_norm:.6f}")
            break
    
        # Generate noisy observation
        noise = 0.1 * np.random.randn(nodes)
        y = D @ (x0 + noise)
        
        # LMS update step
        x = x + mu * (B @ (D @ (y - x)))
    
    return A, Deg, L, eigvals, U, B, D, x0, records

def display_results(A, Deg, L, eigvals, U, B, D, x0, records, nodes):
    # Prepare DataFrames 
    node_labels = list(range(nodes))
    eigvec_labels = [f'u{i}' for i in range(nodes)]
    eigval_labels = [f'λ{i}' for i in range(nodes)]
    
    df_A = pd.DataFrame(A, index=node_labels, columns=node_labels)
    df_Deg = pd.DataFrame(Deg, index=node_labels, columns=node_labels)
    df_L = pd.DataFrame(L, index=node_labels, columns=node_labels)
    df_eigvals = pd.DataFrame(np.round(eigvals, 4), index=[f'λ{i}' for i in range(nodes)], columns=['Eigenvalue'])
    df_U = pd.DataFrame(np.round(U, 4), index=eigvec_labels, columns=eigval_labels)
    df_B = pd.DataFrame(np.round(B, 4), index=node_labels, columns=node_labels)
    df_D = pd.DataFrame(D, index=node_labels, columns=node_labels)
    df_x0 = pd.DataFrame(np.round(x0, 4), index=node_labels, columns=['x0'])
    df_iters = pd.DataFrame(records)
    
    # Display all results
    print("Adjacency Matrix A:\n", df_A, "\n")
    print("Degree Matrix Deg:\n", df_Deg, "\n")
    print("Laplacian Matrix L:\n", df_L, "\n")
    print("EigenValues Λ : \n",df_eigvals,"\n")
    print("Eigenvectors U:\n", df_U, "\n")
    print("Projection Matrix B:\n", df_B, "\n")
    print("Sampling Operator D:\n", df_D, "\n")
    print("True Signal x0:\n", df_x0, "\n")
    print("LMS Estimates over Iterations:\n", df_iters)
    
    return df_iters

def plot_results(df_iters, x0, nodes):
    plt.figure(figsize=(10, 6))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i in range(nodes):
        color = color_cycle[i % len(color_cycle)]  # loop if more nodes than default colors
        plt.plot(df_iters['Iteration'], df_iters[f'x[{i}]'], label=f'Node {i}', color=color)
        plt.axhline(y=x0[i], color=color, linestyle='--', linewidth=1, label=f'x0[{i}] (true)')
    
    plt.title('LMS Signal Estimate per Node Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Signal Value')
    plt.grid(True)
    plt.legend(ncol=2, fontsize='small')
    plt.show()

def plot_graph(A):
    G = nx.from_numpy_array(A)
    
    plt.figure()
    nx.draw_circular(G, with_labels=True)
    plt.title("Graph")
    plt.show()

def main():
    # Define the adjacency matrix and number of nodes
    A = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ])
    
    nodes = 4
    
    # Optional parameters
    F = [0, 1]  # Frequency indices
    mu = 0.5    # Learning rate
    max_iters = 1000
    tolerance = 1e-2
    
    # Run the LMS algorithm
    A, Deg, L, eigvals, U, B, D, x0, records = run_lms_algorithm(
        A, nodes, F, mu, max_iters, tolerance
    )
    
    df_iters = display_results(A, Deg, L, eigvals, U, B, D, x0, records, nodes)
    
    plot_results(df_iters, x0, nodes)

    plot_graph(A)
    
main()
