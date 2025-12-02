import math

# Подынтегральная функция
def f(x):
    return math.sqrt(1 + x**3)

# --------------------------------------------
# Правильный метод трапеций (полный пересчёт)
# --------------------------------------------
def trapezoid(a, b, eps):
    n = 4
    h = (b - a) / n

    # Первое вычисление
    I_prev = h * (0.5*f(a) + 0.5*f(b) + sum(f(a + k*h) for k in range(1, n)))

    while True:
        n *= 2
        h /= 2

        # Полный пересчёт по формуле трапеций
        I_new = h * (0.5*f(a) + 0.5*f(b) + sum(f(a + k*h) for k in range(1, n)))

        if abs(I_new - I_prev) < eps:
            return I_new, n

        I_prev = I_new


# --------------------------------------------
# Правильный метод Симпсона (полный пересчёт)
# --------------------------------------------
def simpson(a, b, eps):
    n = 4       # n должно оставаться чётным
    h = (b - a) / n

    # Функция полного пересчёта по формуле Симпсона
    def Simpson_full(n, h):
        S_odd  = sum(f(a + k*h) for k in range(1, n, 2))
        S_even = sum(f(a + k*h) for k in range(2, n, 2))
        return (h/3) * (f(a) + f(b) + 4*S_odd + 2*S_even)

    # Первое вычисление
    I_prev = Simpson_full(n, h)

    while True:
        n *= 2
        h /= 2

        I_new = Simpson_full(n, h)

        if abs(I_new - I_prev) < eps:
            return I_new, n

        I_prev = I_new

# ---------------------------------------------------
# Основной вызов + сравнение методов
# ---------------------------------------------------
if __name__ == "__main__":
    a = 0.8
    b = 1.762

    for eps in [1e-4, 1e-5]:
        print(f"\n==============================")
        print(f"        eps = {eps}")
        print(f"==============================")

        I_trap, N_trap = trapezoid(a, b, eps)
        I_simp, N_simp = simpson(a, b, eps)

        print(f"Метод трапеций:   I = {I_trap:.10f}, узлов = {N_trap}")
        print(f"Метод Симпсона:   I = {I_simp:.10f}, узлов = {N_simp}")

        # Сравнение результатов
        diff = abs(I_simp - I_trap)
        rel_diff = diff / abs(I_simp) if I_simp != 0 else 0

        print("\n--- Сравнение методов ---")
        print(f"Абсолютная разница:      {diff:.10e}")
        print(f"Относительная разница:   {rel_diff:.10e}")

        if N_simp < N_trap:
            print("Метод Симпсона оказался БЫСТРЕЕ (меньше узлов).")
        elif N_simp > N_trap:
            print("Метод трапеций оказался быстрее — редкий случай!")
        else:
            print("Оба метода использовали одинаковое количество узлов.")

        if diff < eps:
            print("Методы согласуются с необходимой точностью ε.")
        else:
            print("Разница между методами превышает ε — проверить реализацию.")
