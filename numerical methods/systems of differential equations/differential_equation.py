import math

Eps = 0.001
itr_max = 100


# ============================
#   ПРАВАЯ ЧАСТЬ СИСТЕМЫ
# ============================

def func(u, t, n):
    if n == 0:
        if t < Eps:
            return -u[0] * u[1] + 1
        else:
            return -u[0] * u[1] + (math.sin(t) / t)
    elif n == 1:
        return -u[1] * u[1] + (3.5 * t) / (1 + t * t)


# ============================
#      ЯВНЫЙ ЭЙЛЕР
# ============================

def euler_explicit(u, n):
    TauMax = 0.01
    T = 1.0
    t = 0.0

    y = u.copy()

    print("t               u1              u2")

    while t < T:
        tmp = [func(y, t, i) for i in range(n)]

        # Шаг по минимуму (аналог формул 3.11–3.12)
        Tau_candidates = [
            Eps / (abs(tmp[i]) + Eps / TauMax) for i in range(n)
        ]
        Tau = min(Tau_candidates)

        # Обновление решения
        for i in range(n):
            y[i] += Tau * tmp[i]

        t += Tau

        print(f"{t:12.6f} {y[0]:14.6f} {y[1]:14.6f}")


# ============================
#     ФУНКЦИИ ДЛЯ НЕЯВНОГО
# ============================

def f1(uk1, uk, t, Tau):
    return uk1[0] - uk[0] - Tau * (-uk1[0] * uk1[1] + (0 if t < 1e-9 else math.sin(t) / t))


def f2(uk1, uk, t, Tau):
    return uk1[1] - uk[1] - Tau * (-uk1[1] * uk1[1] + (3.125 * t) / (1 + t * t))


# Численная производная
def differential(f, uk1, uk, t, Tau, var_index):
    dx = 1e-9
    D = uk1.copy()
    D[var_index] += dx
    return (f(D, uk, t, Tau) - f(uk1, uk, t, Tau)) / dx


# Метод Гаусса
def gauss(A):
    n = len(A)
    for i in range(n):
        # Поиск максимального элемента
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        if abs(A[max_row][i]) < 1e-12:
            return None

        A[i], A[max_row] = A[max_row], A[i]

        # Нормировка
        pivot = A[i][i]
        A[i] = [x / pivot for x in A[i]]

        # Обнуление ниже
        for r in range(i + 1, n):
            factor = A[r][i]
            A[r] = [A[r][j] - factor * A[i][j] for j in range(n + 1)]

    # Обратный ход
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = A[i][-1] - sum(A[i][j] * x[j] for j in range(i + 1, n))
    return x


# Метод Ньютона
def newton_step(yk1, yk, t, Tau):
    for _ in range(itr_max):
        # Якобиан и вектор −F
        J = [
            [
                differential(f1, yk1, yk, t, Tau, 0),
                differential(f1, yk1, yk, t, Tau, 1),
                -f1(yk1, yk, t, Tau)
            ],
            [
                differential(f2, yk1, yk, t, Tau, 0),
                differential(f2, yk1, yk, t, Tau, 1),
                -f2(yk1, yk, t, Tau)
            ]
        ]

        A = [row[:] for row in J]

        delta = gauss(A)
        if delta is None:
            return None

        yk1 = [yk1[i] + delta[i] for i in range(2)]

        # Ошибка
        b1 = max(abs(f1(yk1, yk, t, Tau)), abs(f2(yk1, yk, t, Tau)))
        b2 = max(abs(d) if abs(d) < 1 else abs(d / yk1[i]) for i, d in enumerate(delta))

        if b1 < 1e-9 and b2 < 1e-9:
            break

    return yk1


# ============================
#   НЕЯВНЫЙ ЭЙЛЕР
# ============================

def euler_implicit(u, n):
    T = 1.0
    t = 0.0

    TauMax = 0.01
    Tau = TauMin = 0.01

    y = u.copy()
    y_minus = u.copy()
    y_plus = u.copy()

    print("t               u1              u2")

    while t < T:

        while True:
            t_new = t + Tau
            y_try = newton_step(y_plus.copy(), y, t, Tau)

            if y_try is None:
                Tau /= 2
                continue

            # Оценка EPS_k
            Eps_k = 0
            for k in range(n):
                Eps_k = -(Tau / (Tau + TauMin)) * (
                    y_try[k] - y[k] - Tau * (y[k] - y_minus[k]) / TauMin
                )

            if abs(Eps_k) <= Eps:
                y_plus = y_try
                break

            Tau /= 2

        # Квазиоптимальный шаг (можете заменить трёхзонным)
        Tau_next = math.sqrt(Eps / abs(Eps_k)) * Tau
        Tau_next = min(Tau_next, TauMax)

        # Печать
        print(f"{t:12.6f} {y[0]:14.6f} {y[1]:14.6f}")

        # Сдвиг
        y_minus = y.copy()
        y = y_plus.copy()

        TauMin = Tau
        Tau = Tau_next
        t = t_new


# ============================
#            MAIN
# ============================

def main():
    u = [0.0, -0.412]
    method = int(input("1 - явный, 2 - неявный: "))

    if method == 1:
        euler_explicit(u, 2)
    else:
        euler_implicit(u, 2)


if __name__ == "__main__":
    main()
