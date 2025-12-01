import dataclasses

import numpy as np
import matplotlib.pyplot as plt

from point import Point


@dataclasses.dataclass
class LinearRegResult:
    mean: Point[float]
    y_predict: np.ndarray


# 1. Генерация данных
def generate_data(seed: int = 89) -> Point[np.ndarray]:
    np.random.seed(seed)

    n = 500
    x_height = np.random.normal(0, 1, n)
    eps = np.random.normal(0, 1, n)
    y_weight = 15 * x_height - 17 + eps

    print("Таблица роста и веса:")
    for i in range(5):
        print(f"Человек {i + 1}: Рост = {x_height[i]:.2f}, Вес = {y_weight[i]:.2f}")

    return Point(x_height, y_weight)


# 2. Строим саму модель и считаем значения
def build_linear_regression(data: Point[np.ndarray]) -> LinearRegResult:
    mean_x = np.mean(data.x)
    mean_y = np.mean(data.y)
    numerator = np.sum((data.x - mean_x) * (data.y - mean_y))
    denominator = np.sum((data.x - mean_x) ** 2)
    b1 = numerator / denominator
    b0 = mean_y - b1 * mean_x

    y_predict = b0 + b1 * data.x

    return LinearRegResult(
        mean=Point(mean_x, mean_y),
        y_predict=y_predict,
    )


# 2. График
def plot_linear_regression(data: Point[np.ndarray], res: LinearRegResult) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(data.x, data.y, label='Исходные данные')
    sort_idx = np.argsort(data.x)
    plt.plot(data.x[sort_idx], res.y_predict[sort_idx], color='red', label='Линия регрессии')
    plt.xlabel('Рост (стандартизированный)')
    plt.ylabel('Вес')
    plt.title('Линейная регрессия: Вес от Роста')
    plt.legend()
    plt.show(block=False)
    plt.savefig('linear_regression.png')


# 3. Оценка R^2
def calc_and_print_results(data: Point[np.ndarray], res: LinearRegResult) -> None:
    # Вычисление RSS, RSE, R^2
    rss = np.sum((data.y - res.y_predict) ** 2)
    tss = np.sum((data.y - res.mean.y) ** 2)
    r2 = 1 - (rss / tss) if tss != 0 else 0
    rse = np.sqrt(rss / (len(data.y) - 2))

    print(f"RSS: {rss:.4f}")
    print(f"RSE: {rse:.4f}")
    print(f"R^2: {r2:.4f}")
