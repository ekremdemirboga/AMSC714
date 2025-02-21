import numpy as np
import matplotlib.pyplot as plt

def exact_solution(x, b):
    r1 = (b + np.sqrt(b**2 + 4)) / 2
    r2 = (b - np.sqrt(b**2 + 4)) / 2
    C1 = (2 * b - 2 - 2 * b * np.exp(r2)) / (np.exp(r1) - np.exp(r2))
    C2 = 2 * b - C1
    return C1 * np.exp(r1 * x) + C2 * np.exp(r2 * x) + 2 * x - 2 * b

def finite_difference_centered(N, b):
    h = 1 / N
    x = np.linspace(0, 1, N+1)
    A = np.zeros((N-1, N-1))
    F = np.zeros(N-1)
    for i in range(1, N):
        if i == 1:
            A[i-1, i-1] = 2 + h**2
            A[i-1, i] = -1 + (b * h) / 2
        elif i == N-1:
            A[i-1, i-2] = -1 - (b * h) / 2
            A[i-1, i-1] = 2 + h**2
        else:
            A[i-1, i-2] = -1 - (b * h) / 2
            A[i-1, i-1] = 2 + h**2
            A[i-1, i] = -1 + (b * h) / 2
        F[i-1] = 2 * h**2 * x[i]
    U = np.linalg.solve(A, F)
    U = np.concatenate(([0], U, [0]))
    return x, U

def finite_difference_upwind(N, b):
    h = 1 / N
    x = np.linspace(0, 1, N+1)
    A = np.zeros((N-1, N-1))
    F = np.zeros(N-1)
    for i in range(1, N):
        if b > 0:
            if i == 1:
                A[i-1, i-1] = 2 + b * h + h**2
                A[i-1, i] = -1
            elif i == N-1:
                A[i-1, i-2] = -1 - b * h
                A[i-1, i-1] = 2 + b * h + h**2
            else:
                A[i-1, i-2] = -1 - b * h
                A[i-1, i-1] = 2 + b * h + h**2
                A[i-1, i] = -1
        else:
            if i == 1:
                A[i-1, i-1] = 2 - b * h + h**2
                A[i-1, i] = -1 + b * h
            elif i == N-1:
                A[i-1, i-2] = -1
                A[i-1, i-1] = 2 - b * h + h**2
            else:
                A[i-1, i-2] = -1
                A[i-1, i-1] = 2 - b * h + h**2
                A[i-1, i] = -1 + b * h
        F[i-1] = 2 * h**2 * x[i]
    U = np.linalg.solve(A, F)
    U = np.concatenate(([0], U, [0]))
    return x, U

b_values = [0, 100]
k_values = range(10)
errors_centered = {b: [] for b in b_values}
errors_upwind = {b: [] for b in b_values}
h_values = []

for k in k_values:
    N = 5 * 2**k
    h = 1 / N
    h_values.append(h)
    x = np.linspace(0, 1, N+1)
    for b in b_values:
        u_exact = exact_solution(x, b)
        _, U_centered = finite_difference_centered(N, b)
        error_centered = np.max(np.abs(u_exact - U_centered))
        errors_centered[b].append(error_centered)
        _, U_upwind = finite_difference_upwind(N, b)
        error_upwind = np.max(np.abs(u_exact - U_upwind))
        errors_upwind[b].append(error_upwind)

plt.figure(figsize=(12, 6))
for b in b_values:
    plt.loglog(h_values, errors_centered[b], 'o-', label=f"Centered Differences (b = {b})")
    plt.loglog(h_values, errors_upwind[b], 's--', label=f"Upwind Differences (b = {b})")
plt.xlabel("Mesh size (h)")
plt.ylabel("Maximum norm error")
plt.title("Error vs. Mesh Size (Log-Log Plot)")
plt.legend()
plt.grid(which="both", linestyle="--")
plt.show()

for h in [1/20, 1/80]:
    plt.figure(figsize=(14, 6))
    for i, b in enumerate(b_values):
        N = int(1 / h)
        x = np.linspace(0, 1, N+1)
        u_exact = exact_solution(x, b)
        x_centered, U_centered = finite_difference_centered(N, b)
        x_upwind, U_upwind = finite_difference_upwind(N, b)
        plt.subplot(1, 2, i+1)
        plt.plot(x, u_exact, label="Exact Solution", linewidth=2)
        plt.plot(x_centered, U_centered, 'o-', label="Centered Differences")
        plt.plot(x_upwind, U_upwind, 's--', label="Upwind Differences")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title(f"Exact and Computed Solutions (h = {h}, b = {b})")
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()