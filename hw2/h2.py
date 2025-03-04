import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def exact_solution(x, b):
    """
    Computes the exact solution of -u'' + b*u' + u = 2x with u(0) = u(1) = 0.
    """
    r1 = (b + np.sqrt(b**2 + 4)) / 2
    r2 = (b - np.sqrt(b**2 + 4)) / 2
    C1 = (2 * b - 2 - 2 * b * np.exp(r2)) / (np.exp(r1) - np.exp(r2))
    C2 = 2 * b - C1
    return C1 * np.exp(r1 * x) + C2 * np.exp(r2 * x) + 2 * x - 2 * b
def exact_solution_derivative(x, b):
    """
    Computes the derivative of the exact solution of -u'' + b*u' + u = 2x with u(0) = u(1) = 0.
    """
    r1 = (b + np.sqrt(b**2 + 4)) / 2
    r2 = (b - np.sqrt(b**2 + 4)) / 2
    C1 = (2 * b - 2 - 2 * b * np.exp(r2)) / (np.exp(r1) - np.exp(r2))
    C2 = 2 * b - C1
    return C1 * r1 * np.exp(r1 * x) + C2 * r2 * np.exp(r2 * x) + 2

def fem_2point(N, b=0):
    """
    Solves -u'' + b*u' + u = 2x on (0,1) with u(0) = u(1) = 0
    using finite element method with piecewise linear elements.
    
    Parameters:
        N (int): Number of elements.
        b (float): Convection coefficient.
    
    Returns:
        x (array): Mesh points.
        u (array): Finite element solution at mesh points.
        h (float): Mesh size.
    """
    # Step 1: Mesh setup
    x = np.linspace(0, 1, N+1)  # Uniform mesh, N+1 nodes
    h = x[1] - x[0]  # Mesh size

    # Step 2: Matrix Assembly
    K = (1/h) * (np.diag([2] * (N-1)) - np.diag([1] * (N-2), -1) - np.diag([1] * (N-2), 1))
    M = (h/6) * (np.diag([4] * (N-1)) + np.diag([1] * (N-2), -1) + np.diag([1] * (N-2), 1))
    C = (1/2) * (np.diag([0] * (N-1)) + np.diag([-1] * (N-2), -1) + np.diag([1] * (N-2), 1))

    # Step 3: Right-hand side vector (Linear term)
    F = (h/2) * (2 * x[1:-1] + 2 * x[2:])

    # Final matrix assembly (Ax = F)
    A = K + M + b * C

    # Step 4: Solve linear system
    u = np.zeros(N+1)  # Include boundary points (zero Dirichlet)
    u[1:-1] = spla.spsolve(sp.csr_matrix(A), F)

    return x, u, h

# Step 5: Compute Errors in H1 and L∞ Norms
def compute_errors(b=0):
    N_values = [5,10, 20, 40, 80, 160,320,640]
    H1_errors = []
    Linf_errors = []

    for N in N_values:
        x, u, h = fem_2point(N, b)
        u_exact = exact_solution(x, b)
        
        # Compute L∞ norm error: max |u - u_h|
        Linf_error = np.max(np.abs(u - u_exact))

        # Compute H1 norm error: sqrt(∫ |u' - u'_h|² dx)
        u_exact_prime = exact_solution_derivative(x,b)
        u_h_prime = np.diff(u) / h  # FEM approximate derivative
        H1_error = np.sqrt(np.sum((u_exact_prime[:-1] - u_h_prime)**2) * h)

        H1_errors.append(H1_error)
        Linf_errors.append(Linf_error)

    return N_values, H1_errors, Linf_errors

# Step 6: Plot Errors
def plot_errors(b=0):
    N_values, H1_errors, Linf_errors = compute_errors(b)
    # plt.figure(figsize=(10, 5))
    # H1 error plot
    plt.loglog(N_values, H1_errors, '-o', label="H1 Error")
    plt.loglog(N_values, Linf_errors, '-o', label="L∞ Error")
    # plt.loglog(N_values, [1/N for N in N_values], '--', label="O(h)")
    plt.xlabel("N (Number of elements)")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Error vs. Mesh Size")
    plt.grid()
    plt.tight_layout()
    plt.show()

# Step 7: Plot FEM Solutions for Different Mesh Sizes
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
b=0
plot_errors(b)
plot_solutions(b)
