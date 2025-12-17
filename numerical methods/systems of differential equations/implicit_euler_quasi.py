import numpy as np
import math

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


def func(t, u):
    u1, u2 = u
    term1 = 1.0 if abs(t) < 1e-9 else math.sin(t) / t
    du1 = -u1 * u2 + term1
    du2 = -u2 ** 2 + (A_PARAM * t) / (1 + t ** 2)
    return np.array([du1, du2])


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


def implicit_euler_quasi():
    t = T_START
    u = U0.copy()
    u_prev = U0.copy()
    tau = TAU_MIN
    steps = 0

    print("\n=== НЕЯВНЫЙ МЕТОД ЭЙЛЕРА (КВАЗИОПТИМАЛЬНАЯ) ===")
    print(f"{'t':<10} | {'u1':<12} | {'u2':<12} | {'tau':<10}")

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

        print(f"{t:<10.4f} | {u[0]:<12.6f} | {u[1]:<12.6f} | {tau:<10.6f}")

        if eps_loc > 1e-14:
            tau = tau * math.sqrt(EPS / eps_loc)

        tau = min(max(tau, TAU_MIN), TAU_MAX)

    print(f"Всего шагов: {steps}")
    return u, steps


if __name__ == "__main__":
    implicit_euler_quasi()
