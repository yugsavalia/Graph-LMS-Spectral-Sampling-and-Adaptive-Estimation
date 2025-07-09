# Adaptive LMS on Graphs

An implementation of an adaptive Least‑Mean‑Squares (LMS) estimator for signals defined over graphs. This repository contains:

- A Python implementation of the adaptive LMS algorithm for graph signals  
- Greedy graph‑sampling strategies to select which vertices to observe  
- Utilities to visualize convergence and the underlying graph structure  


## Topic

Graph signal processing (GSP) extends classical signal processing to data of the nodes of a graph rather than on time or space. If a signal over a graph is **bandlimited**—i.e. its graph Fourier transform is nonzero only on a small set of eigen‑nodes—then it can be reconstructed from samples at just a subset of vertices.  

This project implements the **Adaptive LMS** strategy introduced by Di Lorenzo _et al._, which:  
1. Assumes a bandlimited graph signal and noisy, streaming partial observations at a subset of nodes  
2. Iteratively updates the signal estimate via an LMS‐type recursion projected onto both the sampled vertices and the known frequency support  
3. Adapts the sampling set over time when the bandwidth is unknown, using greedy selection  

The core ideas and theoretical guarantees (mean‐square error bounds, sampling‐set design) are drawn from “Adaptive Least Mean Squares Estimation of Graph Signals” (Di Lorenzo _et al._, 2016).  


## Features

- **Graph Construction & Visualization**  
  - Define arbitrary undirected graphs via adjacency matrices  
  - Plot the graph with NetworkX  

- **Adaptive LMS Algorithm**  
  - Projected LMS recursion `x[n+1] = x[n] + μ · B D (y – x[n])` where  
      - `B`: Projects onto the graph‐frequency support, defined as
          B = U . Σ_F . U^T
        where:
        - U : Graph Fourier basis (eigenvectors of the Laplacian)
        - Σ_F : Diagonal matrix with ones at the frequency indices in the bandlimited set F, zeros elsewhere

      - `D`: Samples only the selected vertices, defined as a **diagonal matrix** with:
          -  D_{ii} = 1  if node i is selected for sampling
          -  D_{ii} = 0  otherwise

  - Convergence checking with tolerance and maximum iterations  

- **Greedy Sampling Strategies**  
  - **Max‑λmin**: Selects the sampling set that **maximizes the minimum nonzero eigenvalue (λ₊ₘᵢₙ) of the matrix B D B**
  - This approach improves the reliability and accuracy of the algorithm by choosing a sampling set that makes the reconstruction process stable and well-behaved.

- **Visualization & Results**  
  - Plotted estimated signal values over iterations and compared them with true signals to assess convergence


## Results

- **Fast convergence** on small graphs (e.g. 4–50 nodes), achieving tolerance 1e‑2 within a few hundred iterations.  
 
## Future Work

- **Implementation of the remaining algorithms**: Extend the project to include the other sampling strategies from the original paper, such as Min of MSD and Max‑Det, and adaptive bandwidth estimation using different thresholding methods (Lasso, Garotte, Hard-thresholding).
- **Comparative testing on real-life data**: Benchmark and compare the performance of all implemented algorithms on real-world graph signal datasets (e.g., sensor network data, traffic networks, or social network graphs) to validate theoretical guarantees in practical scenarios.



