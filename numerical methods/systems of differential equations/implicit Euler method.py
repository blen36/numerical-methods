import numpy as np
import math

# ============================
#      ПАРАМЕТРЫ ЗАДАЧИ
# ============================
OMEGA = 25
A_PARAM = 3.125
U0 = np.array([0.0, -0.412])
T_END = 1.0

# Параметры для неявного метода
EPS = 1e-3
TAU_MIN = 1e-3
TAU_MAX = 0.01


def func(t, u):
    u1, u2 = u
    term1 = 1.0 if abs(t) < 1e-9 else math.sin(t) / t
    du1 = -u1 * u2 + term1
    du2 = -u2 ** 2 + (A_PARAM * t) / (1 + t ** 2)
    return np.array([du1, du2])


# Аналитический Якобиан (для Ньютона)
def get_jacobian(t, u):
    u1, u2 = u
    # df1/du1 = -u2, df1/du2 = -u1
    # df2/du1 = 0,   df2/du2 = -2*u2
    return np.array([
        [-u2, -u1],
        [0.0, -2 * u2]
    ])


def solve_implicit(strategy):
    """
    strategy: 'quasi' (квазиоптимальный) или 'three_zone' (три зоны)
    """
    t = 0.0
    u = U0.copy()
    tau = TAU_MAX  # Начальный шаг

    print(f"\n=== НЕЯВНЫЙ ЭЙЛЕР: {strategy.upper()} ===")
    print(f"{'t':<10} | {'u1':<12} | {'u2':<12} | {'tau':<10}")

    steps = 0
    attempts = 0  # Общее число попыток шагов

    while t < T_END - 1e-9:
        if t + tau > T_END:
            tau = T_END - t

        accepted = False
        while not accepted:
            attempts += 1

            # 3. Вычислить f(t_j, y_j) - нужно для оценки погрешности (3.16)
            f_curr = func(t, u)

            # 4. Решить систему методом Ньютона
            # Уравнение: y_next = u + tau * f(t+tau, y_next)
            # F(y_next) = y_next - u - tau * f(t+tau, y_next) = 0

            # Начальное приближение - явный шаг
            y_next = u + tau * f_curr

            for _ in range(10):  # Ньютон
                F_val = y_next - u - tau * func(t + tau, y_next)
                J_newton = np.eye(2) - tau * get_jacobian(t + tau, y_next)

                try:
                    delta = np.linalg.solve(J_newton, -F_val)
                except np.linalg.LinAlgError:
                    break

                y_next += delta
                if np.max(np.abs(delta)) < 1e-6:
                    break

            # 5. Оценка погрешности по формуле (3.16) из методички
            # E = -0.5 * (y_{j+1} - y_j - tau * f(t_j, y_j))
            epsilon_vec = -0.5 * (y_next - u - tau * f_curr)
            max_err = np.max(np.abs(epsilon_vec))

            # 6. Проверка точности
            if max_err > EPS:
                tau /= 2  # Шаг уменьшаем, повторяем расчет
                if tau < TAU_MIN:
                    tau = TAU_MIN
                    accepted = True  # Вынуждены принять
            else:
                accepted = True

        # Шаг принят
        steps += 1
        t += tau
        u = y_next

        if steps % 10 == 0 or t >= T_END - 1e-9:
            print(f"{t:<10.4f} | {u[0]:<12.6f} | {u[1]:<12.6f} | {tau:<10.6f}")

        # 7. Выбор следующего шага
        if strategy == 'quasi':
            # Формула (3.17): tau_new = tau * sqrt(eps / err)
            if max_err == 0:
                factor = 2.0
            else:
                factor = math.sqrt(EPS / max_err)
            factor = min(factor, 2.0)  # Ограничиваем рост
            tau = tau * factor

        elif strategy == 'three_zone':
            # Формула (3.18)
            if max_err < EPS / 4:
                tau *= 2
            elif max_err > EPS:
                tau /= 2  # Хотя мы выше уже обработали > EPS, это для следующего шага
            # Иначе (в зоне EPS/4 ... EPS) шаг не меняем

        tau = min(tau, TAU_MAX)
        tau = max(tau, TAU_MIN)

    print(f"Итоговых шагов по времени: {steps}")
    print(f"Всего вычислений (с учетом возвратов): {attempts}")
    return u, steps


if __name__ == "__main__":
    solve_implicit('quasi')
    solve_implicit('three_zone')