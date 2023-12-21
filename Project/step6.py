# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:44:07 2023

@author: ZC
"""

import numpy as np
from scipy.io import mmread
from scipy.sparse.linalg import splu, bicgstab
from scipy.linalg import qr, solve
import matplotlib.pyplot as plt
import time
import scipy.sparse.linalg as sla

def relative_residual(A, x, f):
    residual_norm = np.linalg.norm(A.dot(x) - f)
    f_norm = np.linalg.norm(f)
    return residual_norm / f_norm

def ilu_preconditioned_bicgstab(A, f, threshold=1e-5, max_iterations=1000):
    # ILU preconditioning
    B = sla.spilu(A, drop_tol=1e-12, fill_factor=1)
    Mz = lambda r: B.solve(r)
    Minv = sla.LinearOperator(A.shape, matvec=Mz)

    # Solve the system using ILU preconditioned BiCGSTAB
    x, info = sla.bicgstab(A, f, tol=threshold, maxiter=max_iterations, M=Minv)

    return x

def block_kaczmarz(A, f, K, max_iter=1000):
    x = np.zeros(A.shape[1])
    A_dense = A.toarray()  # Convert to dense array
    for _ in range(max_iter):
        for i in range(0, A.shape[0], K):
            A_block = A_dense[i:i+K, :]
            f_block = f[i:i+K]
            x += np.dot(A_block.T, np.linalg.solve(A_block.dot(A_block.T), f_block - A_block.dot(x)))
    return x

def block_cimmino(A, f, K, max_iter=1000):
    x = np.zeros(A.shape[1])
    A_dense = A.toarray()  # Convert to dense array
    for _ in range(max_iter):
        for i in range(0, A.shape[0], K):
            A_block = A_dense[i:i+K, :]
            f_block = f[i:i+K]
            delta_i = np.linalg.solve(A_block.dot(A_block.T), f_block - A_block.dot(x))
            x += np.dot(A_block.T, delta_i)
    return x


# Load sparse matrix A and ndarray f
A = mmread('poisson2D.mtx')
f = mmread('poisson2D_b.mtx')

# Make sure f is a 1D array
f = np.squeeze(np.array(f))

# Convert A to CSR format for efficient spsolve
A_csr = A.tocsr()

orderings = ['NATURAL', 'MMD_ATA', 'COLAMD', 'BICGSTAB']
max_iterations = 1000
threshold = 1e-5

for ordering in orderings:
    print(f"\nSolving with {ordering} ordering:")
    
    # Measure the time
    start_time = time.time()

    if ordering == 'NATURAL':
        # Solve the system using sparse LU factorization with the specified ordering
        lu = splu(A_csr, permc_spec=ordering)
        x = lu.solve(f)
    elif ordering == 'BICGSTAB':
        # Solve the system using ILU preconditioned BiCGSTAB
        x = ilu_preconditioned_bicgstab(A_csr, f, threshold, max_iterations)
    else:
        # QR factorization
        Q, R = qr(A.toarray())
        y = np.dot(Q.T, f)
        x = solve(R, y)

    end_time = time.time()

    # Print the solution and the time taken for solving
    print("Time taken:", end_time - start_time, "seconds")

    # Compute and print the relative residual
    rel_residual = relative_residual(A_csr, x, f)
    print("Relative Residual:", rel_residual)

    # Check if the solution is accurate enough based on the threshold
    if rel_residual < threshold:
        print("The solution is accurate (below threshold).")
    else:
        print("The solution may not be accurate enough (above threshold).")

    # You can also plot the solution if needed
    plt.plot(x)
    plt.title(f'Solution x with {ordering} ordering')
    plt.show()

# Define block sizes
block_sizes = [2, 4, 6, 8]

for K in block_sizes:
    print(f"\nRunning Block Kaczmarz with K={K}:")
    start_time = time.time()
    x_kaczmarz = block_kaczmarz(A_csr, f, K)
    end_time = time.time()
    print("Time taken:", end_time - start_time, "seconds")
    rel_residual_kaczmarz = relative_residual(A_csr, x_kaczmarz, f)
    print("Relative Residual:", rel_residual_kaczmarz)

    print(f"\nRunning Block Cimmino with K={K}:")
    start_time = time.time()
    x_cimmino = block_cimmino(A_csr, f, K)
    end_time = time.time()
    print("Time taken:", end_time - start_time, "seconds")
    rel_residual_cimmino = relative_residual(A_csr, x_cimmino, f)
    print("Relative Residual:", rel_residual_cimmino)
