import numpy as np

def input_matrix_and_vector():
    n = int(input("Введите размерность матрицы n: "))
    A = []
    print("Введите матрицу A (построчно через пробел):")
    for _ in range(n):
        row = list(map(float, input().split()))
        A.append(row)
    A = np.array(A)

    print("Введите вектор b:")
    b = np.array(list(map(float, input().split())))
    return A, b


def residual(A, x, b):
    F = A @ x - b
    delta = np.max(np.abs(F))
    return F, delta



def relative_error(x_exact, x_approx):
    return np.linalg.norm(x_exact - x_approx) / np.linalg.norm(x_exact)


def gauss_solve(A, b):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    # Прямой ход
    for k in range(n - 1):
        # Выбор главного элемента
        max_row = np.argmax(abs(A[k:n, k])) + k
        if A[max_row, k] == 0:
            raise ValueError("Матрица вырождена!")
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


def is_symmetric(A, tol=1e-9):
    return np.allclose(A, A.T, atol=tol)


def ldlt_decomposition(A):
    if not is_symmetric(A):
        raise ValueError("Матрица несимметрична, LDL^T факторизация невозможна")

    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros((n, n))

    for j in range(n):
        D[j, j] = A[j, j] - sum(L[j, k] ** 2 * D[k, k] for k in range(j))
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - sum(L[i, k] * L[j, k] * D[k, k] for k in range(j))) / D[j, j]

    return L, D


def ldlt_solve(A, b):
    L, D = ldlt_decomposition(A)
    n = len(b)

    # Решаем Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Решаем Dz = y
    z = y / np.diag(D)

    # Решаем L^T x = z
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - np.dot(L[i + 1:, i], x[i + 1:]))

    return x


# =================== ОСНОВНОЙ ИНТЕРФЕЙС ===================
if __name__ == "__main__":
    print("Выберите метод:")
    print("1 - Метод Гаусса")
    print("2 - LDL^T-факторизация")
    choice = int(input("Ваш выбор: "))

    A, b = input_matrix_and_vector()

    if choice == 1:
        x = gauss_solve(A.copy(), b.copy())
        print("Решение методом Гаусса:", x)
    elif choice == 2:
        x = ldlt_solve(A.copy(), b.copy())
        print("Решение методом LDL^T:", x)
    else:
        print("Некорректный выбор")

    print("Вектор невязки:", residual(A, x, b))