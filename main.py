# %% Импортируем нужные библиотеки и определяем классы и методы
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import pandas as pd
import math
from scipy.stats import chi2

class Model:  # Модель из предыдущих лабораторных работ

    def __init__(self):
        self.amount_tests = 300
        self.x_max = 1
        self.x_min = -1
        self.x1 = []
        self.x2 = []
        self.signal = []  # сигнал
        self.response = []  # отклик
        self.variance = []  # дисперсия
        self.theta = np.array([1, 4, 0.001, 0.001, 4])  # параметры модели
        self.theta_mnk = []  # Оценка теты по МНК
        self.theta_general_mnk = []  # Оценка теты по обобщенному МНК
        self.func = lambda x1, x2: 1 + 4 * x1 + 0.001 * x2 + 0.001 * x1 ** 2 + 4 * x2 ** 2
        self.experiment_matrix = []


class Calculator:

    @staticmethod
    def compute_signal(model: Model):  # Вычисление сигнала - незашумленного отклика
        signal = [model.func(model.x1[i], model.x2[i])
                  for i in range(model.amount_tests)]
        return np.array(signal)

    @staticmethod
    def compute_variance(model):  # Вычисление дисперсии (взвешенная сумма квадратов факторов)
        result = np.array([0.5 * model.x1[i] ** 2 + 0.5 * model.x2[i] ** 2 for i in range(model.amount_tests)])
        return result

    @staticmethod
    def compute_response(model, error):  # вычисление зашумленного отклика
        return model.signal + error

    @staticmethod
    def general_mnk(model):  # Обобщенный метод наименьших квадратов
        matrix_V = np.diag(model.variance)
        general_mnk_eval = np.matmul(np.matmul(np.matmul(np.linalg.inv(
            np.matmul(np.matmul(model.experiment_matrix.T, np.linalg.inv(matrix_V)), model.experiment_matrix)),
            model.experiment_matrix.T), np.linalg.inv(matrix_V)), model.response)
        return general_mnk_eval

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
            np.array([1 for _ in range(model.amount_tests)]),
            model.x1,
            model.x2,
            np.array([x1 ** 2 for x1 in model.x1]),
            np.array([x2 ** 2 for x2 in model.x2])
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


class DataGenerator:

    @staticmethod
    def generate_couple(x_min, x_max, amount_tests):  # Генерация значений регрессоров
        x1 = np.random.uniform(x_min, x_max, amount_tests)
        x2 = np.random.uniform(x_min, x_max, amount_tests)
        return x1, x2

    @staticmethod
    def generate_error(standard_deviation, number_tests) -> float:  # генерация случайной ошибки
        error = np.random.normal(0, standard_deviation, number_tests)  # стандартное отклонение - sqrt(variance)
        return error

