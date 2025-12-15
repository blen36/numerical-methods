import numpy as np
import math

# ===================== ПАРАМЕТРЫ =====================
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


# ===================== НЬЮТОН ДЛЯ НЕЯВНОГО ШАГА =====================
def newton_step(u_prev, t_next, tau):
    u = u_prev.copy()

    for _ in range(NEWTON_MAX_ITER):
        F = u - u_prev - tau * func(t_next, u)

        u1, u2 = u
        df1_du1 = -u2
        df1_du2 = -u1
        df2_du1 = 0.0
        df2_du2 = -2.0 * u2

        J = np.array([
            [1.0 - tau * df1_du1, -tau * df1_du2],
            [-tau * df2_du1, 1.0 - tau * df2_du2]
        ])

        delta = np.linalg.solve(J, -F)
        u = u + delta

        if np.max(np.abs(delta)) < NEWTON_EPS:
            return u

    raise RuntimeError("Метод Ньютона не сошёлся")


# ===================== НЕЯВНЫЙ МЕТОД ЭЙЛЕРА =====================
def implicit_euler():
    t = T_START
    u = U0.copy()
    u_prev = U0.copy()

    tau = TAU_MIN
    steps = 0

    print("\n=== НЕЯВНЫЙ МЕТОД ЭЙЛЕРА ===")
    print(f"{'t':<10} | {'u1':<12} | {'u2':<12} | {'tau':<10}")

    while t < T_END - 1e-9:
        steps += 1

        if t + tau > T_END:
            tau = T_END - t

        # неявный шаг
        u_next = newton_step(u, t + tau, tau)

        # оценка локальной погрешности
        if steps > 1:
            eps_loc = np.max(np.abs(
                (u_next - u) / tau - (u - u_prev) / tau
            )) * tau / 2.0
        else:
            eps_loc = 0.0

        # контроль шага
        if eps_loc > EPS and tau > TAU_MIN:
            tau = tau / 2.0
            continue

        # принимаем шаг
        t += tau
        u_prev = u.copy()
        u = u_next

        if steps % 5 == 0 or t >= T_END - 1e-9:
            print(f"{t:<10.4f} | {u[0]:<12.6f} | {u[1]:<12.6f} | {tau:<10.6f}")

        # пересчёт шага
        if eps_loc > 1e-14:
            tau_new = tau * math.sqrt(EPS / eps_loc)
        else:
            tau_new = TAU_MAX

        tau = min(max(tau_new, TAU_MIN), TAU_MAX)

    print(f"Всего итераций: {steps}")
    return u


# ===================== ЗАПУСК =====================
if __name__ == "__main__":
    implicit_euler()
