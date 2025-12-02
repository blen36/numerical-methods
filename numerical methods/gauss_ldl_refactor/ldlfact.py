import numpy as np

def ldl_factorization(A):

    n = A.shape[0]
    L = np.eye(n)  # единичная матрица
    D = np.zeros((n, n))

    for j in range(n):
        # d_jj
        s = 0
        for k in range(j):
            s += (L[j, k] ** 2) * D[k, k]
        D[j, j] = A[j, j] - s

        # элементы столбца L
        for i in range(j+1, n):
            s = 0
            for k in range(j):
                s += L[i, k] * L[j, k] * D[k, k]
            L[i, j] = (A[i, j] - s) / D[j, j]

    return L, D


def ldl_solve(A, b):
    """
    Решение Ax = b через LDL^T факторизацию.
    """
    L, D = ldl_factorization(A)

    # 1. Ly = b (прямой ход)
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # 2. Dz = y
    z = y / np.diag(D)

    # 3. L^T x = z (обратный ход)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = z[i] - np.dot(L[i+1:, i], x[i+1:])

    return x


# ---------- Вариант 21 ----------
def build_matrix_and_vector(l1, l2, l3):
    A = np.array([
        [2*l1 + 4*l2,   2*(l1 - l2),     2*(l1 - l2)],
        [2*(l1 - l2),   2*l1 + l2 + 3*l3, 2*l1 + l2 - 3*l3],
        [2*(l1 - l2),   2*l1 + l2 - 3*l3, 2*l1 + l2 + 3*l3]
    ], dtype=float)

    b = np.array([
        -4*l1 - 2*l2,
        -4*l1 + l2 + 9*l3,
        -4*l1 + l2 - 9*l3
    ], dtype=float)

    return A, b


# ---------- Тест с параметрами λ1=1, λ2=1e3, λ3=1e6 ----------
l1, l2, l3 = 1, 1e3, 1e6
A, b = build_matrix_and_vector(l1, l2, l3)

print("Матрица A:\n", A)
print("\nВектор b:\n", b)

x = ldl_solve(A, b)
print("\nРешение x:", x)

# ---------- Проверка невязки ----------
F = A @ x - b
delta = np.max(np.abs(F))
print("\nВектор невязки F:", F)
print("Норма Δ =", delta)
