import math
import numpy as np
import matplotlib.pyplot as plt

H = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
P = np.array([760.0, 674.8, 598.0, 528.9, 466.6, 410.6, 360.2])
N = len(H)
m = 1

Y = np.log10(P)

POWER_X = [sum(H**k) for k in range(0, 2*m + 1)]

SUM_X = np.zeros((m+1, m+1))
for i in range(m+1):
    for j in range(m+1):
        SUM_X[i, j] = POWER_X[i + j]

PRAW = np.zeros(m+1)
for i in range(m+1):
    PRAW[i] = sum(Y * (H**i))

def gauss_solve(A, b):
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = len(b)

    for i in range(n):
        pivot = i + np.argmax(np.abs(A[i:, i]))
        if A[pivot, i] == 0:
            raise ValueError("Singular matrix")
        A[[i, pivot]] = A[[pivot, i]]
        b[[i, pivot]] = b[[pivot, i]]

        div = A[i, i]
        A[i] /= div
        b[i] /= div

        for j in range(i+1, n):
            factor = A[j, i]
            A[j] -= factor * A[i]
            b[j] -= factor * b[i]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i] - np.dot(A[i, i+1:], x[i+1:])
    return x

A_linear = gauss_solve(SUM_X, PRAW)

log10_a = A_linear[0]
b = A_linear[1]
a = 10**log10_a

P_approx = a * (10**(b * H))

deviations = P_approx - P
S2 = sum(deviations**2) / (N - m - 1)
s = math.sqrt(S2)

print("Model: p = a * 10^(b * h)")
print(f"a = {a:.6f}, b = {b:.6f}\n")

print(" h     P      φ(h)     dev")
for hi, pi, fi in zip(H, P, P_approx):
    print(f"{hi:3.1f}  {pi:7.1f}  {fi:8.1f}  {fi-pi:8.2f}")

print(f"\nОстаточная дисперсия S^2 = {S2:.6f}")
print(f"Среднеквадратичное отклонение s = {s:.6f}")

plt.figure(figsize=(12, 6))
plt.scatter(H, P, color='blue', label='f(x) — исходные данные')

H_smooth = np.linspace(min(H), max(H), 300)
P_smooth = a * (10**(b * H_smooth))
plt.plot(H_smooth, P_smooth, color='red', linewidth=2, label='φ(x) — аппроксимация')

plt.title('Аппроксимация p = a · 10^{b·h}')
plt.xlabel('h')
plt.ylabel('p')
plt.grid(True)
plt.legend()
plt.show()
