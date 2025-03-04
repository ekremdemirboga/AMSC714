import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Define the exact solution
def exact_solution(x, b):
    """
    Computes the exact solution of -u'' + b*u' + u = 2x with u(0) = u(1) = 0.
    """
    r1 = (b + np.sqrt(b**2 + 4)) / 2
    r2 = (b - np.sqrt(b**2 + 4)) / 2
    C1 = (2 * b - 2 - 2 * b * np.exp(r2)) / (np.exp(r1) - np.exp(r2))
    C2 = 2 * b - C1
    return C1 * np.exp(r1 * x) + C2 * np.exp(r2 * x) + 2 * x - 2 * b

# Define the derivative of the exact solution
def exact_solution_derivative(x, b):
    """
    Computes the derivative of the exact solution of -u'' + b*u' + u = 2x with u(0) = u(1) = 0.
    """
    r1 = (b + np.sqrt(b**2 + 4)) / 2
    r2 = (b - np.sqrt(b**2 + 4)) / 2
    C1 = (2 * b - 2 - 2 * b * np.exp(r2)) / (np.exp(r1) - np.exp(r2))
    C2 = 2 * b - C1
    return C1 * r1 * np.exp(r1 * x) + C2 * r2 * np.exp(r2 * x) + 2

# Define the graded mesh
def graded_mesh(N):
    """
    Generates a graded mesh with points concentrated near x = 0.
    """
    return np.array([(i / N) ** 4 for i in range(N + 1)])

# FEM solver with piecewise linear elements
def fem_2point(N, b=0,beta=1.01):
    """
    Solves -u'' + b*u' + u = 2x on (0,1) with u(0) = u(1) = 0
    using finite element method with piecewise linear elements.
    
    Parameters:
        N (int): Number of elements.
        b (float): Convection coefficient.
    
    Returns:
        x (array): Mesh points.
        u (array): Finite element solution at mesh points.
        h (array): Mesh sizes.
    """
    # Step 1: Mesh setup (graded mesh)
    x = 1 - np.power((N - np.arange(N+1))/N, beta)
    h = np.diff(x)  # Mesh sizes (non-uniform)

    # Step 2: Matrix Assembly (adjust for non-uniform mesh)
    K = sp.diags([1 / h[:-1]], [0], shape=(N-1, N-1)) + \
        sp.diags([-1 / h[:-1]], [1], shape=(N-1, N-1)) + \
        sp.diags([-1 / h[1:]], [-1], shape=(N-1, N-1)) + \
        sp.diags([1 / h[1:]], [0], shape=(N-1, N-1))
    
    M = sp.diags([h[:-1] / 3], [0], shape=(N-1, N-1)) + \
        sp.diags([h[:-1] / 6], [1], shape=(N-1, N-1)) + \
        sp.diags([h[1:] / 6], [-1], shape=(N-1, N-1)) + \
        sp.diags([h[1:] / 3], [0], shape=(N-1, N-1))
    
    C = sp.diags([-0.5], [0], shape=(N-1, N-1)) + \
        sp.diags([0.5], [1], shape=(N-1, N-1)) + \
        sp.diags([-0.5], [-1], shape=(N-1, N-1)) + \
        sp.diags([0.5], [0], shape=(N-1, N-1))

    # Step 3: Right-hand side vector (Linear term)
    F = (h[1:] / 2) * (2 * x[1:-1] + 2 * x[2:])

    # Final matrix assembly (Ax = F)
    A = K + M + b * C

    # Step 4: Solve linear system
    u = np.zeros(N + 1)  # Include boundary points (zero Dirichlet)
    u[1:-1] = spla.spsolve(A.tocsr(), F)

    return x, u, h

# Compute errors in H1 and L∞ norms
def compute_errors(b=0):
    N_values = [20, 40, 80, 160, 320, 640]
    H1_errors = []
    Linf_errors = []

    for N in N_values:
        x, u, h = fem_2point(N, b)
        u_exact = exact_solution(x, b)
        
        # Compute L∞ norm error: max |u - u_h|
        Linf_error = np.max(np.abs(u - u_exact))

        # Compute H1 norm error: sqrt(∫ |u' - u'_h|² dx)
        u_exact_prime = exact_solution_derivative(x, b)
        u_h_prime = np.diff(u) / h  # FEM approximate derivative
        H1_error = np.sqrt(np.sum((u_exact_prime[:-1] - u_h_prime)**2 * h))
        print(H1_error)

        H1_errors.append(H1_error)
        Linf_errors.append(Linf_error)

    return N_values, H1_errors, Linf_errors

# Plot errors
def plot_errors(b=0):
    N_values, H1_errors, Linf_errors = compute_errors(b)
    plt.figure(figsize=(10, 5))
    
    # H1 error plot
    plt.loglog(N_values, H1_errors, '-o', label="H1 Error")
    plt.loglog(N_values, Linf_errors, '-o', label="L∞ Error")
    plt.loglog(N_values, [1 / N**2 for N in N_values], '--', label="O(h^2)")
    plt.loglog(N_values, [1 / N for N in N_values], '--', label="O(h)")
    
    plt.xlabel("N (Number of elements)")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Error vs. Mesh Size")
    plt.grid()
    plt.tight_layout()
    plt.show()

# Plot FEM solutions for different mesh sizes
def plot_solutions(b=0):
    plt.figure()
    for N in [20, 80, 320]:
        x, u, _ = fem_2point(N, b)
        plt.plot(x, u, label=f'N={N}')
    plt.plot(x,exact_solution(x,b),'--',label='exact')
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("FEM Solutions for different mesh sizes")
    plt.show()

# Run the scripts
b = 100  # Advection-dominated problem
plot_errors(b)
plot_solutions(b)