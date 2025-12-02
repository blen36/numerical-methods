import math

# Подынтегральная функция
def f(x):
    return math.sqrt(1 + x**3)

# --------------------------------------------
# Метод Симпсона на сгущающихся сетках
# --------------------------------------------
def simpson(a, b, eps):
    n = 4                      # должно быть чётным
    h = (b - a) / n

    # Первое вычисление
    def Simpson_full(n, h):
        S_odd = sum(f(a + k*h) for k in range(1, n, 2))
        S_even = sum(f(a + k*h) for k in range(2, n, 2))
        return (h/3) * (f(a) + f(b) + 4*S_odd + 2*S_even)

    I_prev = Simpson_full(n, h)

    # Сгущение сетки
    while True:
        n *= 2
        h /= 2
        I_new = Simpson_full(n, h)

        if abs(I_new - I_prev) < eps:
            return I_new, n

        I_prev = I_new

# Пример запуска
if __name__ == "__main__":
    a = 0.8
    b = 1.762
    eps = 1e-5

    I_val, N = simpson(a, b, eps)
    print(f"Simpson method: I = {I_val}, nodes = {N}")
