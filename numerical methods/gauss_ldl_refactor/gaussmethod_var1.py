import numpy as np

# ---------- Метод Гаусса ----------
def gauss_solve(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # Прямой ход
    for k in range(n):
        # выбор главного элемента в столбце
        max_row = k
        max_val = abs(A[k, k])
        for i in range(k+1, n):
            if abs(A[i, k]) > max_val:
                max_val = abs(A[i, k])
                max_row = i

        # перестановка строк
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        # проверка на нулевой pivot
        if abs(A[k, k]) < 1e-12:
            raise ZeroDivisionError(f"Нулевой ведущий элемент в строке {k}")

        # обнуление под диагональю
        for i in range(k+1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = np.dot(A[i, i+1:], x[i+1:])
        x[i] = (b[i] - s) / A[i, i]

    return x

# ---------- Невязка ----------
def residual(A, x, b):
    F = A @ x - b
    delta = np.max(np.abs(F))
    return F, delta


# ---------- Ввод варианта ----------
variant = int(input("Введите номер варианта (1 или 21): "))

if variant == 1:
    # Вариант 1
    A = np.array([[6, 13, -17],
                  [13, 29, -38],
                  [-17, -38, 50]], dtype=float)
    b = np.array([2, 4, -5], dtype=float)

elif variant == 21:
    # Вариант 21
    l1 = float(input("Введите λ1: "))
    l2 = float(input("Введите λ2: "))
    l3 = float(input("Введите λ3: "))

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
else:
    raise ValueError("Нужно ввести вариант 1 или 21")

# ---------- Решение исходной системы ----------
x_hat = gauss_solve(A.copy(), b.copy())
print("\nРешение исходной системы x_hat:", x_hat)

# ---------- Вспомогательная система ----------
b_aux = A @ x_hat  # правая часть: A*x_hat
x_tilde = gauss_solve(A.copy(), b_aux.copy())
print("Решение вспомогательной системы x_tilde:", x_tilde)

# ---------- Относительная погрешность ----------
numerator = np.max(np.abs(x_tilde - x_hat))
denominator = np.max(np.abs(x_hat))
if denominator < 1e-12:
    relative_error = numerator  # чтобы избежать деления на 0
else:
    relative_error = numerator / denominator

print("\nОценка относительной погрешности:")
print(f"Числитель     : {numerator:.10e}")
print(f"Знаменатель   : {denominator:.10e}")
print(f"Отн. погрешн. δ: {relative_error:.10e}")

# ---------- Проверка невязки ----------
F, delta = residual(A, x_hat, b)
print("\nВектор невязки F:", F)
print("Норма Δ =", delta)
