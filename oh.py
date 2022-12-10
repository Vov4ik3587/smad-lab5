import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import pandas as pd
import math
from scipy.stats import chi2

m = 7  # количество параметров
n = 100  # количество экспериментов
c = [1, 1, 1, 1, 1, 1, 1]  # истинные значения параметров


def function(x1, x2, x3, x4, x5, x6):
    return 1 + x1 + x2 + x3 + x4 + x5 + x6


def ocenca_p(X_T, X, y):
    X1 = X_T @ X
    X2 = np.linalg.inv(X1)
    X3 = X2 @ X_T
    return X3 @ y


def generate_X(x1, x2, x3, x4, x5, x6, n_):
    X = np.empty((n_, m), dtype="float32")

    for i in range(n_):
        X[i][0] = 1
        X[i][1] = x1[i]
        X[i][2] = x2[i]
        X[i][3] = x3[i]
        X[i][4] = x4[i]
        X[i][5] = x5[i]
        X[i][6] = x6[i]
    return X


def graph(lambdas, ev, RSS):
    xtick = []

    for i in range(500):
        xtick.append(0.02 * i)

    plt.figure(figsize=(10, 7))
    plt.xticks(xtick, size=7)
    plt.plot(lambdas, RSS, 'green')
    plt.ylabel('RSS')
    plt.xlabel('Лямбда')
    plt.title('Зависимость RSS от лямбда')
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.xticks(xtick, size=7)
    plt.plot(lambdas, ev, 'blue')
    plt.ylabel('Евклидова норма оценок')
    plt.xlabel('Лямбда')
    plt.title('Изменение евклидовой нормы оценок')
    plt.show()


def sort(x1, x2, y):
    # имеющиеся наблюдения поставили в соответсвие (скомбинировали)
    # такой скомбинированный массив сортируем по столбцу, отражающему х1**2
    data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    data = data.sort_values(by='x1', key=lambda x: x ** 2)

    # обновляем массивы с учетом сортировки
    x1 = data['x1'].to_list()
    x2 = data['x2'].to_list()
    y = data['y'].to_list()
    return x1, x2, y


def take_part(x1, i, num):
    x_new = np.empty(num, dtype=float)

    for j in range(num):
        x_new[j] = x1[i + j]
    return x_new


def generate():
    # ----- задаём значения факторов в диапазоне [-1, 1] для n экспериментов
    x1 = np.empty(n)
    x2 = np.random.uniform(-1, 1, n)
    x3 = np.random.uniform(-0.5, 0.5, n)
    x4 = np.random.uniform(-1, 1, n)
    x5 = np.random.uniform(-2, 2, n)
    x6 = np.random.uniform(-5, 5, n)

    # для создания значений вектора x1 используем x2, x3 и мини погрешность
    for i in range(n):
        x1[i] = x2[i] + x3[i] + np.random.normal(0, 0.1)
    # определяем незашумленный отклик
    u_ = 0;
    u = [function(x1[i], x2[i], x3[i], x4[i], x5[i], x6[i]) for i in range(n)]
    for i in range(n):
        u_ += u[i]

    # определим мощность сигнала
    w = 0  # мощность сигнала

    for i in range(n):
        w += (u[i] - u_) ** 2 / (n - 1)

    # вычисление дисперсии при доле p=5%
    d = 0.05 * w

    #  значение помехи для n экспериментов
    e = np.random.normal(0, np.sqrt(d), n)  # шум

    #  значение зашумленного отклика
    y = [0 for i in range(n)]  # зашумлённый отклик

    for i in range(n):
        y[i] = u[i] + e[i]

        # строим матрицу X
        X = generate_X(x1, x2, x3, x4, x5, x6, n)

    return x1, x2, x3, x4, x5, x6, X, u, y


def MNK(X, y):
    p_ = ocenca_p(X.transpose(), X, y)
    y_ = X @ p_
    e_ = y - y_
    d_ = e_.transpose() @ e_ / n
    return p_, e_, d_


def MinMaxLambd(X):
    # собственные числа матрицы
    l = np.linalg.eigvals(X)
    return min(l), max(l)


def main():
    # ----- строим модель
    # x1, x2, ..., x6 - значения факторов
    # X - матрица значений функции при разных значениях факторов
    # u - незашумлённый отклик
    # y - зашумленный отклик

    x1, x2, x3, x4, x5, x6, X, u, y = generate()
    file = open('lab5.txt', 'w')

    trace = np.trace(X.T @ X)

    detX1 = np.linalg.det(X.T @ X)  # определитель информационной матрицы
    detX = np.linalg.det(X.T @ X / trace)  # Определитель информационной матрицы, нормированной
    file.write("\nОпределитель информационной матрицы X^TX: " + str(detX1) + "\n\n")
    file.write("\nОпределитель информационной матрицы X^TX/trace(X^TX): " + str(detX) + "\n\n")

    # минимальное и максимальное собственные числа матрицы
    # lmin - минимальное, lmax - максимальное собственные числа
    lmin, lmax = MinMaxLambd(X.T @ X)

    file.write("Минимальное собственное число матрицы: " + str(lmin))
    file.write("Максимальное собственное число матрицы: " + str(lmax) + "\n\n")
    # мера обусловленности матрицы  (Нейман-Голдстейн)
    NeimGold = lmax / lmin

    file.write("Мера обусловленности: " + str(NeimGold) + "\n\n")

    # максимальная парная сопряжённость

    R = np.zeros((m, m))  # матрица сопряжённости размерностью с количесвто факторов
    maxRij = 0  # максимальная парная сопряжённость

    # вычисляем max(rij) и заполняем R
    for i in range(m):
        for j in range(m):
            R[i][j] = np.dot(X.T[i], X.T[j]) / (np.sqrt(np.dot(X.T[i], X.T[i]) * np.dot(X.T[j], X.T[j])))
            if maxRij < R[i][j] and i != j:
                maxRij = R[i][j]

    file.write(str(R))
    file.write("\nМаксимальная парная сопряжённость: " + str(maxRij) + "\n\n")

    # максимальная сопряжённость

    Rt = [0 for i in range(m)]
    maxRi = 0  # максимальная сопряжённость
    R1 = np.linalg.inv(R)  # обратная матрица

    for i in range(0, m):
        Rt[i] = np.sqrt(1.0 - 1.0 / R1[i][i])

    maxRi = np.max(Rt)

    file.write(str(Rt))
    file.write("\nМаксимальная сопряжённость: " + str(maxRi) + "\n\n")

    # ридж-оценки

    lm = 0.005  # параметр лямбда малая
    lambdas = [lm]  # массив для хранения промежуточных значений лямбда
    RSS = []  # остаточная сумма квадратов
    ev = []  # евклидова норма оценок параметров

    XTX = X.T @ X
    L = np.diagonal(XTX)  # взяли диагональку
    L = np.eye(len(L)) * L  # диагональка и нули

    while lm < 0.5:
        p = np.linalg.inv(XTX + lm * L) @ X.T @ y  # ридж-оценки при текущем значении лямбда
        yt = X @ p.T
        RSSh = sum([(yt[i] - y[i]) ** 2 for i in range(len(y))])
        RSS.append(RSSh)  # добавить в конец списка
        evh = np.linalg.norm(p)
        ev.append(evh)

        file.write("\nЛямбда = " + str(lm))
        file.write("Ридж оценки:\n" + str(p))
        file.write("RSS: " + str(RSSh))
        file.write("Евклидова норма: " + str(evh))

        lm += 0.05
        lambdas.append(lm)

        lambdas.pop()  # удаляем лишний элемент (на него сработал выход значит он нам не нужен)

    # строим графики
    graph(lambdas, ev, RSS)

    # метод главных компонентов

    Xzv = []

    for i in range(len(p)):
        Xzv.append(X.T[i] - X.T[i].mean())  # центрирование переменных

    Xzv = np.array(Xzv)
    Xzv = Xzv.T
    Xxt = Xzv.T @ Xzv  # выборочная ковариационная матрица вектора

    # находим собственные числа и вектора матрицы
    lambdas, V = np.linalg.eig(Xxt)

    # собственные значения с незначительным вкладом
    lambdasr = []
    for i in range(len(lambdas)):
        lambdasr.append(lambdas[i] / sum(lambdas))

    file.write("\nВклады параметров: " + str(lambdasr))
    V = np.delete(V, m - 1, 0)
    V = np.delete(V, m - 2, 0)
    # определяем матрицу значений главных компонент
    Z = Xzv @ V.T

    # оценка регрессии на главные компоненты
    yzv = np.array(y) - np.array(y).mean()
    b = np.linalg.inv(Z.T @ Z) @ Z.T @ yzv

    # оценка параметров
    p2 = V.T @ b
    # RSS
    RSSk = sum([((X @ p.T)[i] - y[i]) ** 2 for i in range(len(y))])
    # евклидова норма оценок параметров
    EVk = np.linalg.norm(p2)

    file.write("\nМГК оценки: \n")
    file.write(str(p2))
    file.write("RSS: " + str(RSSk))
    file.write("Евклидова норма: " + str(EVk))
    file.close()


main()
