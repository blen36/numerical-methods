import numpy as np
import math

# ===================== ОБЩИЕ ПАРАМЕТРЫ =====================
OMEGA = 25
A_PARAM = 2.5 + OMEGA / 40.0

U0 = np.array([0.0, -0.412])

T_START = 0.0
T_END = 1.0

EPS = 1e-3
TAU_MIN = 1e-3
TAU_MAX = 0.01

NEWTON_EPS = 1e-9
NEWTON_MAX_ITER = 50


# ===================== ПРАВАЯ ЧАСТЬ =====================
def func(t, u):
    u1, u2 = u
    term1 = 1.0 if abs(t) < 1e-9 else math.sin(t) / t

    du1 = -u1 * u2 + term1
    du2 = -u2 ** 2 + (A_PARAM * t) / (1 + t ** 2)

    return np.array([du1, du2])


# ===================== ЯВНЫЙ МЕТОД ЭЙЛЕРА =====================
def explicit_euler_method():
    t = T_START
    u = U0.copy()
    steps = 0

    while t < T_END - 1e-9:
        steps += 1

        f_val = func(t, u)
        max_f = np.max(np.abs(f_val))

        if max_f < 1e-9:
            tau = TAU_MAX
        else:
            tau = EPS / max_f

        tau = min(tau, TAU_MAX)

        if t + tau > T_END:
            tau = T_END - t

        u = u + tau * f_val
        t += tau

    return u, steps

def gauss_solve(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    for k in range(n):
        max_row = max(range(k, n), key=lambda i: abs(A[i, k]))
        if abs(A[max_row, k]) < 1e-12:
            raise ZeroDivisionError("Нулевой pivot")

        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x

# ===================== НЬЮТОН ДЛЯ НЕЯВНОГО МЕТОДА =====================
def newton_step(u_prev, t_next, tau):
    u = u_prev.copy()

    for _ in range(NEWTON_MAX_ITER):
        F = u - u_prev - tau * func(t_next, u)

        u1, u2 = u
        J = np.array([
            [1.0 + tau * u2, tau * u1],
            [0.0, 1.0 + 2.0 * tau * u2]
        ])

        delta = gauss_solve(J, -F)
        u = u + delta

        if np.max(np.abs(delta)) < NEWTON_EPS:
            return u

    raise RuntimeError("Метод Ньютона не сошёлся")


# ===================== НЕЯВНЫЙ МЕТОД ЭЙЛЕРА =====================
def implicit_euler_quasi():
    t = T_START
    u = U0.copy()
    u_prev = U0.copy()
    tau = TAU_MIN
    steps = 0

    while t < T_END - 1e-9:
        if t + tau > T_END:
            tau = T_END - t

        u_next = newton_step(u, t + tau, tau)
        steps += 1

        if steps > 1:
            eps_loc = np.max(np.abs(
                (u_next - u) / tau - (u - u_prev) / tau
            )) * tau / 2
        else:
            eps_loc = 0.0

        if eps_loc > EPS and tau > TAU_MIN:
            tau /= 2
            continue

        t += tau
        u_prev = u.copy()
        u = u_next

        if eps_loc > 1e-14:
            tau = tau * math.sqrt(EPS / eps_loc)

        tau = min(max(tau, TAU_MIN), TAU_MAX)

    return u, steps


# ===================== СРАВНЕНИЕ МЕТОДОВ =====================
def compare_methods():
    u_explicit, steps_explicit = explicit_euler_method()
    u_implicit, steps_implicit = implicit_euler_quasi()

    print("\n================= СРАВНЕНИЕ МЕТОДОВ =================")
    print(f"{'Метод':<30} | {'u1(T)':<12} | {'u2(T)':<12} | {'Шаги'}")
    print("-" * 70)
    print(f"{'Explicit Euler method':<30} | {u_explicit[0]:<12.6f} | {u_explicit[1]:<12.6f} | {steps_explicit}")
    print(f"{'Implicit Euler method':<30} | {u_implicit[0]:<12.6f} | {u_implicit[1]:<12.6f} | {steps_implicit}")

    diff = np.linalg.norm(u_explicit - u_implicit, ord=np.inf)
    print("-" * 70)
    print(f"Максимальное отличие решений: {diff:.3e}")


# ===================== ЗАПУСК =====================
if __name__ == "__main__":
    compare_methods()
