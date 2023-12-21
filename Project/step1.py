import numpy as np
from scipy.io import mmread
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time

# Load sparse matrix A and ndarray f
A = mmread('poisson2D.mtx')
f = mmread('poisson2D_b.mtx')

# Make sure f is a 1D array
f = np.squeeze(np.array(f))

# Solve the linear system Ax = b
start_time = time.time()
x = spsolve(A, f)
end_time = time.time()

# Print the solution and the time taken for solving
print("Solution x:", x)
print("Time taken:", end_time - start_time, "seconds")

# You can also plot the solution if needed
plt.plot(x)
plt.title('Solution x')
plt.show()
