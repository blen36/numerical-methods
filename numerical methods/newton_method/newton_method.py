import math
import numpy as np

def f_vec(x):
    x1, x2 = x[0], x[1]
    arg = 1.0 + (x1 + x2) / 5.0
    if arg <= 0:
        # защита: возвращаем большой невязкой, чтобы метод понял проблему
        return np.array([1e6, 1e6], dtype=float)
    f1 = math.log(arg) - math.sin(x2 / 3.0) - x1 + 1.1
    f2 = math.cos((x1 * x2) / 6.0) - x2 + 0.5
    return np.array([f1, f2], dtype=float)

def J_analytic(x):
    x1, x2 = x[0], x[1]
    denom = 5.0 + x1 + x2
    if abs(denom) < 1e-16:
        denom = 1e-16
    df1_dx1 = 1.0 / denom - 1.0
    df1_dx2 = 1.0 / denom - (1.0 / 3.0) * math.cos(x2 / 3.0)
    df2_dx1 = - (x2 / 6.0) * math.sin((x1 * x2) / 6.0)
    df2_dx2 = - (x1 / 6.0) * math.sin((x1 * x2) / 6.0) - 1.0
    return np.array([[df1_dx1, df1_dx2],
                     [df2_dx1, df2_dx2]], dtype=float)

def J_numeric(x, M):
    n = len(x)
    J = np.zeros((n, n), dtype=float)
    for j in range(n):
        xj = x[j]
        dx = M * abs(xj) if abs(xj) > 0 else M
        if dx == 0:
            dx = 1e-12
        xp = x.copy(); xm = x.copy()
        xp[j] += dx
        xm[j] -= dx
        fp = f_vec(xp)
        fm = f_vec(xm)
        # центральная разность
        J[:, j] = (fp - fm) / (2.0 * dx)
    return J

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

def newton_method(x0, J_mode='analytic', M=0.01, eps1=1e-9, eps2=1e-9, max_iter=100):
    x = np.array(x0, dtype=float)
    print()
    if J_mode == 'analytic':
        print("Метод Ньютона — аналитический Якоби")
    else:
        print(f"Метод Ньютона — численный Якоби, M = {M}")

    print(" k |      x1       |      x2       |     f1      |     f2      |     d1      |     d2      |   ||F||inf  |  ||d||inf  ")
    print("-"*120)

    for k in range(1, max_iter+1):
        F = f_vec(x)
        if J_mode == 'analytic':
            J = J_analytic(x)
        else:
            J = J_numeric(x, M)

        # ПРОВЕРКА НА ВЫРОЖДЕННОСТЬ ЯКОБИАНА
        det = np.linalg.det(J)
        if abs(det) < 1e-10:
            print(f"{k:2d} | {x[0]: .10f} | {x[1]: .10f} | {F[0]: .4e} | {F[1]: .4e} | {'---':>11} | {'---':>11} | {np.max(np.abs(F)): .4e} | {'---':>9}")
            print("-" * 120)
            print(f"ПРЕРВАНО: Вырожденный Якобиан (det = {det:.2e}) на итерации {k}")
            return x, k

        # Решаем систему J * delta = -F
        delta = gauss_solve(J, -F)


        normF = np.max(np.abs(F))
        x = x + delta
        if x < 1:
            normDelta = np.max(np.abs(delta))
        elif x >= 1:
            normDelta = np.max(np.abs(delta)/x)

        print(f"{k:2d} | {x[0]: .10f} | {x[1]: .10f} | {F[0]: .4e} | {F[1]: .4e} | {delta[0]: .4e} | {delta[1]: .4e} | {normF: .4e} | {normDelta: .4e}")

        # критерий остановки: одновременно ||F|| < eps1 и ||delta|| < eps2
        if normF < eps1 and normDelta < eps2:
            x = x + delta
            print("-"*120)
            print(f"Сошлось за {k} итераций. Решение: x1 = {x[0]:.10f}, x2 = {x[1]:.10f}, ||F||inf = {normF:.3e}")
            return x, k



    print("-"*120)
    print(f"Не сошлось за {max_iter} итераций. Последнее приближение: x1 = {x[0]:.10f}, x2 = {x[1]:.10f}, ||F||inf = {np.max(np.abs(f_vec(x))):.3e}")
    return x, max_iter

def main():
    x0 = [8.0, 8.0]
    eps1 = 1e-9
    eps2 = 1e-9
    max_iter = 50

    newton_method(x0, J_mode='analytic', M=None, eps1=eps1, eps2=eps2, max_iter=max_iter)

    for M in [0.01, 0.05, 0.1]:
        newton_method(x0, J_mode='numeric', M=M, eps1=eps1, eps2=eps2, max_iter=max_iter)

if __name__ == "__main__":
    main()
