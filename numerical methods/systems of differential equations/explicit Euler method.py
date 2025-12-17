import numpy as np
import math

OMEGA = 25
A_PARAM = 2.5 + OMEGA / 40.0
U0 = np.array([0.0, -0.412])
T_START = 0.0
T_END = 1.0

EPS = 1e-3
TAU_MAX = 0.01

def func(t, u):
    u1, u2 = u
    term1 = 1.0 if abs(t) < 1e-9 else math.sin(t) / t

    du1 = -u1 * u2 + term1
    du2 = -u2 ** 2 + (A_PARAM * t) / (1 + t ** 2)
    return np.array([du1, du2])

def explicit_euler():
    t = T_START
    u = U0.copy()
    tau_max = TAU_MAX

    print("\n=== ЯВНЫЙ МЕТОД ЭЙЛЕРА ===")
    print(f"{'t':<10} | {'u1':<12} | {'u2':<12} | {'tau':<10}")

    steps = 0
    while t < T_END - 1e-9:
        steps += 1

        f_val = func(t, u)

        max_f = np.max(np.abs(f_val))
        if max_f < 1e-9:
            tau = tau_max
        else:
            tau = EPS / max_f

        tau = min(tau, tau_max)

        if t + tau > T_END:
            tau = T_END - t

        u_next = u + tau * f_val

        t += tau
        u = u_next

        if steps % 10 == 0 or t >= T_END - 1e-9:
            print(f"{t:<10.4f} | {u[0]:<12.6f} | {u[1]:<12.6f} | {tau:<10.6f}")

    print(f"Всего итераций: {steps}")
    return u


if __name__ == "__main__":
    explicit_euler()