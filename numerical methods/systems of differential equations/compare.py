import numpy as np
import math

# ==========================================
# 1. Определение системы и параметров задачи
# ==========================================

# Вариант 1, Омега = 25
OMEGA = 25
A_PARAM = 2.5 + OMEGA / 40.0  # 3.125
T_START = 0.0
T_END = 1.0
U0 = np.array([0.0, -0.412])  # Начальные условия


# Функция правой части: f(t, u)
def func(t, u):
    u1, u2 = u

    # Обработка неопределенности sin(t)/t при t=0
    if abs(t) < 1e-9:
        term1 = 1.0
    else:
        term1 = math.sin(t) / t

    du1 = -u1 * u2 + term1
    du2 = -u2 ** 2 + (A_PARAM * t) / (1 + t ** 2)
    return np.array([du1, du2])


# Матрица Якоби для метода Ньютона (нужна для неявного метода)
# J = df/du
def get_jacobian(t, u):
    u1, u2 = u
    # df1/du1 = -u2, df1/du2 = -u1
    # df2/du1 = 0,   df2/du2 = -2*u2
    return np.array([
        [-u2, -u1],
        [0.0, -2 * u2]
    ])


# ==========================================
# 2. Явный метод Эйлера (Алгоритм 3.1)
# ==========================================
def explicit_euler(eps, tau_max):
    """
    Реализация Алгоритма 3.1 из методички .
    """
    t = T_START
    u = U0.copy()

    # Списки для хранения траектории
    trajectory_t = [t]
    trajectory_u = [u]
    iterations = 0

    print(f"\n--- Явный метод Эйлера ---")

    while t < T_END:
        iterations += 1

        # 3. Вычислить вектор f
        f_val = func(t, u)

        # 4. Определить шаг tau по формулам (3.11), (3.12)
        # Условие устойчивости и точности: tau <= eps / max(|f_i|)
        max_f = np.max(np.abs(f_val))
        if max_f < 1e-9:
            tau = tau_max
        else:
            tau = eps / max_f

        # Ограничение сверху
        if tau > tau_max:
            tau = tau_max

        # Чтобы не перешагнуть через конец отрезка
        if t + tau > T_END:
            tau = T_END - t

        # 5. Выполнить шаг: y_next = y + tau * f
        u_next = u + tau * f_val

        # 6. Вычислить t_next
        t_next = t + tau

        # 7. (В алгоритме вывод на печать, мы сохраняем)
        trajectory_t.append(t_next)
        trajectory_u.append(u_next)

        # Обновление для следующего шага
        t = t_next
        u = u_next

        # 8. Проверка окончания (цикл while)

    return iterations, trajectory_t, trajectory_u


# ==========================================
# 3. Неявный метод Эйлера (Алгоритм 3.2)
# ==========================================
def implicit_euler(eps, tau_min, tau_max, step_strategy = "quasi"):
    """
    Реализация Алгоритма 3.2 из методички .
    step_strategy: "quasi" (квазиоптимальный) или "three_zone" (три зоны)
    """
    t = T_START
    u = U0.copy()
    tau = tau_max  # Начальный шаг (п. 2)

    trajectory_t = [t]
    trajectory_u = [u]
    iterations = 0

    print(f"\n--- Неявный метод Эйлера ({step_strategy}) ---")

    while t < T_END:
        # Корректировка последнего шага
        if t + tau > T_END:
            tau = T_END - t

        accepted_step = False

        while not accepted_step:
            iterations += 1  # Считаем попытки шагов

            # 3. Вычислить f(t_j, y_j) - нужно для оценки погрешности (3.16)
            f_curr = func(t, u)

            # 4. Решить методом Ньютона систему (3.14): y - y_prev - tau * f(t_new, y) = 0
            # Начальное приближение для Ньютона: y^(0) = y_prev (или явный Эйлер)
            y_new = u + tau * f_curr  # Используем явный шаг как предиктор

            # Цикл Ньютона
            for _ in range(10):  # Максимум 10 итераций Ньютона
                # F(y) = y_new - u - tau * func(t+tau, y_new)
                F_val = y_new - u - tau * func(t + tau, y_new)

                # Якобиан системы уравнений Ньютона: J_newton = I - tau * J_func
                J_func = get_jacobian(t + tau, y_new)
                J_newton = np.eye(2) - tau * J_func

                # Поправка: delta = - J^(-1) * F
                try:
                    delta = np.linalg.solve(J_newton, -F_val)
                except np.linalg.LinAlgError:
                    break  # Если матрица вырождена

                y_new = y_new + delta
                if np.max(np.abs(delta)) < 1e-6:  # Сходимость Ньютона
                    break

            # 5. Вычислить локальную погрешность E по формуле (3.16)
            # E = -0.5 * (y_next - y_prev - tau * f(t_prev, y_prev))
            error_vec = -0.5 * (y_new - u - tau * f_curr)
            max_error = np.max(np.abs(error_vec))

            # 6. Проверка условия точности
            if max_error > eps:
                # Если ошибка велика, уменьшаем шаг и повторяем (п. 6)
                tau = tau / 2
                if tau < tau_min:
                    tau = tau_min
                    accepted_step = True  # Вынуждены принять минимальный шаг
                    # print("Warning: Step size reached minimum limit.")
            else:
                accepted_step = True

        # Шаг принят
        # 7. Определить новый шаг для следующей итерации

        if step_strategy == "quasi":
            # Квазиоптимальный выбор (3.17)
            # tau_new = tau_old * sqrt(eps / |max_error|)
            if max_error == 0:
                factor = 2.0
            else:
                factor = math.sqrt(eps / max_error)

            # Обычно ограничивают рост шага, чтобы не скакал сильно
            factor = min(factor, 2.0)
            tau_new = tau * factor

        elif step_strategy == "three_zone":
            # Правило трех зон (3.18)
            # Мы уже проверили max_error > eps выше. Здесь max_error <= eps.
            if max_error < eps / 4:
                tau_new = tau * 2  # Зона увеличения шага
            else:
                tau_new = tau  # Зона сохранения шага

        # Ограничение по tau_max
        if tau_new > tau_max:
            tau_new = tau_max

        # 8. (В методичке п.8 дублирует п.6, но по логике мы тут сохраняем)
        # 9. Сохранение/Вывод
        t_next = t + tau
        u = y_new
        t = t_next

        trajectory_t.append(t)
        trajectory_u.append(u)

        # 10. Сдвиг шага
        tau = tau_new

        if t >= T_END - 1e-9:
            break

    return iterations, trajectory_t, trajectory_u


# ==========================================
# 4. Основной блок запуска и сравнения
# ==========================================

# Параметры из задания
EPS_EXPLICIT = 1e-3
TAU_MAX_EXPLICIT = 0.01

EPS_IMPLICIT = 1e-3
TAU_MIN_IMPLICIT = 1e-3
TAU_MAX_IMPLICIT = 0.01

# 1. Решение явным методом
iter_exp, t_exp, u_exp = explicit_euler(EPS_EXPLICIT, TAU_MAX_EXPLICIT)
res_exp = u_exp[-1]

# 2. Решение неявным методом (Квазиоптимальный шаг)
iter_imp_quasi, t_iq, u_iq = implicit_euler(EPS_IMPLICIT, TAU_MIN_IMPLICIT, TAU_MAX_IMPLICIT, step_strategy = "quasi")
res_imp_quasi = u_iq[-1]

# 3. Решение неявным методом (Трехзонный шаг)
iter_imp_3z, t_i3, u_i3 = implicit_euler(EPS_IMPLICIT, TAU_MIN_IMPLICIT, TAU_MAX_IMPLICIT, step_strategy = "three_zone")
res_imp_3z = u_i3[-1]

# ==========================================
# 5. Вывод результатов
# ==========================================

print("\n" + "=" * 60)
print(f"РЕЗУЛЬТАТЫ (Система №1, a={A_PARAM})")
print("=" * 60)

print(f"{'Метод':<40} | {'Итераций':<10} | {'u1(1.0)':<10} | {'u2(1.0)':<10}")
print("-" * 75)

print(f"{'Явный Эйлер':<40} | {iter_exp:<10} | {res_exp[0]:.6f}   | {res_exp[1]:.6f}")
print(
    f"{'Неявный Эйлер (Квазиоптимальный шаг)':<40} | {iter_imp_quasi:<10} | {res_imp_quasi[0]:.6f}   | {res_imp_quasi[1]:.6f}")
print(f"{'Неявный Эйлер (Трехзонный шаг)':<40} | {iter_imp_3z:<10} | {res_imp_3z[0]:.6f}   | {res_imp_3z[1]:.6f}")

print("-" * 75)
print("Анализ:")
if iter_exp > iter_imp_quasi:
    print(f"-> Неявный метод потребовал в {iter_exp / iter_imp_quasi:.1f} раз меньше шагов по времени.")
    print("   Это связано с тем, что явный метод вынужден сильно дробить шаг")
    print("   для выполнения условия устойчивости (3.9), тогда как неявный")
    print("   ограничен только требуемой точностью.")
else:
    print("-> Шаги методов сопоставимы (возможно, система не является жесткой")
    print("   или ограничения tau_max не дают неявному методу разогнаться).")