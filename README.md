# Adaptive LMS on Graphs

An implementation of an adaptive Least‑Mean‑Squares (LMS) estimator for signals defined over graphs. This repository contains:

- A Python implementation of the adaptive LMS algorithm for graph signals  
- Greedy graph‑sampling strategies to select which vertices to observe  
- Utilities to visualize convergence and the underlying graph structure  


## Topic

Graph signal processing (GSP) extends classical signal processing to data “living” on the nodes of a graph rather than on time or space. If a signal over a graph is **bandlimited**—i.e. its graph Fourier transform is nonzero only on a small set of eigen‑modes—then it can be reconstructed from samples at just a subset of vertices.  

This project implements the **Adaptive LMS** strategy introduced by Di Lorenzo _et al._, which:  
1. Assumes a bandlimited graph signal and noisy, streaming partial observations at a subset of nodes  
2. Iteratively updates the signal estimate via an LMS‐type recursion projected onto both the sampled vertices and the known frequency support  
3. Adapts the sampling set over time when the bandwidth is unknown or time‐varying, using sparse thresholding (Lasso, Garotte, or hard‐threshold) plus greedy selection  

The core ideas and theoretical guarantees (mean‐square error bounds, sampling‐set design) are drawn from “Adaptive Least Mean Squares Estimation of Graph Signals” (Di Lorenzo _et al._, 2016).  


## Features

- **Graph Construction & Visualization**  
  - Define arbitrary undirected graphs via adjacency matrices  
  - Plot the graph with NetworkX  

- **Adaptive LMS Algorithm**  
  - Projected LMS recursion `x[n+1] = x[n] + μ · B D (y – x[n])` where  
    - `B` projects onto the graph‐frequency support  
    - `D` samples only selected vertices  
  - Convergence checking with tolerance and maximum iterations  

- **Greedy Sampling Strategies**  
  - **Max‑λmin**: maximize minimum nonzero eigenvalue of Uᴴ D U  

- **Adaptive Bandwidth Estimation**  
  - Sampling‐set updated each iteration to match estimated bandwidth  

- **Visualization & Results**  
  - Final sampling‐set overlays on the graph  


## Results

- **Fast convergence** on small graphs (e.g. 4–50 nodes), achieving tolerance 1e‑2 within a few hundred iterations.  
- **Sampling‐set effectiveness**: Max‑Det and Max‑λmin strategies yield low steady‐state MSD even with $(|S| = |F|)$ minimum samples; Min‑MSD outperforms others once $|S| > |F|$.  
- **Adaptive tracking**: Hard‐thresholding based adaptive algorithm accurately tracks time‐varying bandwidths (switching among 5, 15, 10 GFT modes) with minimal overshoot.  
- **Graph mismatch robustness**: Removing certain edges causes mild performance degradation, illustrating sensitivity of the GFT basis to topology changes.  



