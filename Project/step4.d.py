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
