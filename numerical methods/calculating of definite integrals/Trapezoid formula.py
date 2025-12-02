import math

# Подынтегральная функция
def f(x):
    return math.sqrt(1 + x**3)

# --------------------------
# Метод трапеций со сгущением сетки
# --------------------------
def trapezoid(a, b, eps):
    n = 4                      # начальное разбиение
    h = (b - a) / n

    # Первое вычисление
    I_prev = h * (0.5*f(a) + 0.5*f(b) + sum(f(a + k*h) for k in range(1, n)))

    # Сгущение сетки
    while True:
        n *= 2
        h /= 2

        # Полный пересчёт по определению (методичка НЕ разрешает ускорений)
        I_new = h * (0.5*f(a) + 0.5*f(b) + sum(f(a + k*h) for k in range(1, n)))

        if abs(I_new - I_prev) < eps:
            return I_new, n

        I_prev = I_new


# Пример запуска
if __name__ == "__main__":
    a = 0.8
    b = 1.762
    eps = 1e-5

    I_val, N = trapezoid(a, b, eps)
    print(f"Trapezoid method: I = {I_val},  nodes = {N}")
