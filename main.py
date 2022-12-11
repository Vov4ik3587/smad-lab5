# %% Импортируем нужные библиотеки и определяем классы и методы
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import pandas as pd
import math
from scipy.stats import chi2


# Модель на 7 факторах, 4 из них создают мультиколлинеарность. Масштаб у всех факторов одинаковый
class Model:

    def __init__(self):
        self.amount_tests = 100
        self.x_max = 1
        self.x_min = -1
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.x4 = []
        self.x5 = []
        self.x6 = []
        self.x7 = []
        self.power = 0
        self.signal = []  # сигнал
        self.response = []  # отклик
        self.variance = []  # дисперсия
        self.theta = np.ones(7)  # параметры модели
        self.theta_mnk = []  # Оценка теты по МНК
        self.theta_general_mnk = []  # Оценка теты по обобщенному МНК
        self.func = lambda x1, x2, x3, x4, x5, x6, x7: 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7
        self.experiment_matrix = []


class Calculator:

    @staticmethod
    def compute_signal(model: Model):  # Вычисление сигнала - незашумленного отклика
        signal = [model.func(model.x1[i], model.x2[i], model.x3[i], model.x4[i], model.x5[i], model.x6[i], model.x7[i])
                  for i in range(model.amount_tests)]
        return np.array(signal)

    @staticmethod
    def compute_variance(model):  # Вычисление дисперсии (взвешенная сумма квадратов факторов)
        return model.power * 0.1

    @staticmethod
    def compute_response(model, error):  # вычисление зашумленного отклика
        return np.array(model.signal + error)

    @staticmethod
    def mnk(model):  # Метод наименьших квадратов
        trans_experiment_matrix = model.experiment_matrix.T
        mnk_eval = np.matmul(np.linalg.inv(np.matmul(trans_experiment_matrix, model.experiment_matrix)),
                             trans_experiment_matrix)
        mnk_eval = np.matmul(mnk_eval, model.response)
        return mnk_eval

    @staticmethod
    def compute_experiment_matrix(model):  # Матрица наблюдений X
        experiment_matrix = np.array([
            np.ones(model.amount_tests),
            model.x1,
            model.x2,
            model.x3,
            model.x4,
            model.x5,
            model.x6,
            model.x7,
        ], dtype=object)
        experiment_matrix = np.array([list(i) for i in zip(*experiment_matrix)])
        return experiment_matrix

    @staticmethod
    def sort(model):
        data = pd.DataFrame({'x1': model.x1, 'x2': model.x2, 'y': model.response})
        data = data.sort_values(by='x1', key=lambda x: x ** 2)

        x1 = data['x1'].to_list()
        x2 = data['x2'].to_list()
        y = data['y'].to_list()
        return x1, x2, y

    @staticmethod
    def compute_power(model):
        avg_signal = [
            np.sum(model.signal) / len(model.signal)
            for i in range(len(model.signal))
        ]
        vec_avg_signal = np.array(avg_signal)
        power = np.vdot(model.signal - vec_avg_signal,
                        model.signal - vec_avg_signal) / len(model.signal)
        return power


class DataGenerator:

    @staticmethod
    def generate_couple(x_min, x_max, amount_tests):  # Генерация значений регрессоров
        x1 = np.random.uniform(x_min, x_max, amount_tests)
        x2 = np.random.uniform(x_min, x_max, amount_tests)
        x3 = np.random.uniform(x_min, x_max, amount_tests)
        x4 = np.random.uniform(x_min, x_max, amount_tests)
        x5 = np.random.uniform(x_min, x_max, amount_tests)
        x6 = np.random.uniform(x_min, x_max, amount_tests)
        x7 = x4 + x5 + x6 + np.random.normal(0, 0.1, amount_tests)
        return x1, x2, x3, x4, x5, x6, x7

    @staticmethod
    def generate_error(standard_deviation, number_tests) -> float:  # генерация случайной ошибки
        error = np.random.normal(0, standard_deviation, number_tests)  # стандартное отклонение - sqrt(variance)
        return error


# %% Заполняем модель данными

model = Model()

model.x1, model.x2, model.x3, model.x4, model.x5, model.x6, model.x7 = DataGenerator.generate_couple(
    model.x_min, model.x_max, model.amount_tests)

model.signal = Calculator.compute_signal(model)

model.power = Calculator.compute_power(model)

model.variance = Calculator.compute_variance(model)

error = DataGenerator.generate_error(
    np.sqrt(model.variance), model.amount_tests)

model.response = Calculator.compute_response(model, error)

model.experiment_matrix = Calculator.compute_experiment_matrix(model)

# %% Рассчитаем показатели мультиколлинеарности

# Определитель информационной матрицы X.T X

info_mat = model.experiment_matrix.T @ model.experiment_matrix

det_info_mat = np.linalg.det(info_mat)
det_info_mat_norm = np.linalg.det(info_mat / np.trace(info_mat))

print(f"Определитель информационной матрицы: {det_info_mat}")
print(f"Определитель информационной матрицы, нормированной на след: {det_info_mat_norm}")

# Минимальное собственное число информационной матрицы

lyambda = np.linalg.eigvals(info_mat)
lyambda_min, lyambda_max = np.min(lyambda), np.max(lyambda)

print(f"Минимальное собственное число информационной матрицы: {lyambda_min}")

# Мера обусловленности по Нейману-Голдстейну

measure = lyambda_max / lyambda_min

print(f"Мера обусловленности по Нейману-Голдстейну: {measure}")

# Максимальная парная сопряженность
m = len(model.theta)

R = np.zeros((m, m))

maxRij = 0

for i in range(m):
    for j in range(m):
        R[i][j] = np.dot(model.experiment_matrix.T[i], model.experiment_matrix.T[j]) / (np.sqrt(
            np.dot(model.experiment_matrix.T[i], model.experiment_matrix.T[i]) * np.dot(model.experiment_matrix.T[j],
                                                                                        model.experiment_matrix.T[j])))
        if maxRij < R[i][j] and i != j:
            maxRij = R[i][j]

print(f"Максимальная парная сопряженность: {maxRij}")

# Максимальная парная сопряженность

Rt = np.zeros(m)

maxRi = 0  # максимальная сопряжённость
R1 = np.linalg.inv(R)  # обратная матрица

for i in range(0, m):
    Rt[i] = np.sqrt(1.0 - 1.0 / R1[i][i])

maxRi = np.max(Rt)

print(f"Максимальная сопряженность: {maxRi}")

# %% Найдем ридж оценки

lm = 0.005  # параметр лямбда малая
lambdas = [lm]  # массив для хранения промежуточных значений лямбда
RSS = []  # остаточная сумма квадратов
ev = []  # евклидова норма оценок параметров

L = np.diagonal(info_mat)  # взяли диагональку
L = np.eye(len(L)) * L  # диагональка и нули

while lm < 0.5:  # ищем оптимальный параметр регуляризации
    p = np.linalg.inv(
        info_mat + lm * L) @ model.experiment_matrix.T @ model.response  # ридж-оценки при текущем значении лямбда
    yt = model.experiment_matrix @ p.T
    RSSh = sum([(yt[i] - model.response[i]) ** 2 for i in range(len(model.response))])
    RSS.append(RSSh)
    evh = np.linalg.norm(p)
    ev.append(evh)

    print(f"\nЛямбда = {lm}")
    print(f"Ридж-оценки = {p}")
    print(f"RSS = {RSSh}")
    print(f"Евклидова норма = {evh}")

    lm += 0.05
    lambdas.append(lm)

lambdas.pop()

# %% Строим нужные графики

xtick = [0.02 * i for i in range(500)]

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

# %% Метод главных компонент

Xzv = np.array([model.experiment_matrix.T[i] - model.experiment_matrix.T[i].mean() for i in range(len(p))]).T

Xxt = Xzv.T @ Xzv  # выборочная ковариационная матрица вектора

# находим собственные числа и вектора матрицы
lambdas, V = np.linalg.eig(Xxt)

# собственные значения с незначительным вкладом
lambdasr = []
for i in range(len(lambdas)):
    lambdasr.append(lambdas[i] / sum(lambdas))

print(f"\nВклады параметров: {str(lambdasr)}")

V = np.delete(V, m - 1, 0)
V = np.delete(V, m - 2, 0)
# определяем матрицу значений главных компонент
Z = Xzv @ V.T

# оценка регрессии на главные компоненты
yzv = model.response - model.response.mean()

b = np.linalg.inv(Z.T @ Z) @ Z.T @ yzv

# оценка параметров
p2 = V.T @ b
# RSS
RSSk = np.sum([((model.experiment_matrix @ p.T)[i] - model.response[i]) ** 2 for i in range(len(model.response))])
# евклидова норма оценок параметров
EVk = np.linalg.norm(p2)

print(f"\nМГК оценки: {p2} \n")
print(f"RSS: {RSSk}")
print(f"Евклидова норма: {EVk}")
