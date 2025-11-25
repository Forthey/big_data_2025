import numpy as np
import matplotlib.pyplot as plt

from point import Point


# 4. Смещение значений, чтобы > 0
def modify_data(data: Point[np.ndarray]) -> Point[np.ndarray]:
    x_age = data.x - np.min(data.x) + 1
    y_weight_age = data.y - np.min(data.y) + 1

    print("\nСмещённая таблица возраста и веса:")
    for i in range(5):
        print(f"Человек {i + 1}: Возраст = {x_age[i]:.2f}, Вес = {y_weight_age[i]:.2f}")

    return Point(x_age, y_weight_age)


# 4. Строим саму модель и считаем значения
def build_multiplicative_model(data: Point[np.ndarray]) -> np.ndarray:
    log_x = np.log(data.x)
    log_y = np.log(data.y)

    mean_logx = np.mean(log_x)
    mean_logy = np.mean(log_y)

    numerator_log = np.sum((log_x - mean_logx) * (log_y - mean_logy))
    denominator_log = np.sum((log_x - mean_logx) ** 2)

    b = numerator_log / denominator_log
    loga = mean_logy - b * mean_logx
    a = np.exp(loga)

    log_y_predict = loga + b * log_x
    return np.exp(log_y_predict)


# 4. График
def plot_multiplicative_model(data: Point[np.ndarray], y_predict: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(data.x, data.y, label='Исходные данные')

    sort_idx = np.argsort(data.x)
    plt.plot(data.x[sort_idx], y_predict[sort_idx], color='red', label='Кривая регрессии')

    plt.xlabel('Возраст (сдвинутый)')
    plt.ylabel('Вес (сдвинутый)')
    plt.title('Мультипликативная регрессия: Вес ~ Возраст')
    plt.legend()
    plt.show(block=False)
    plt.savefig('multiplicative_model.png')


# 4. Оценка R^2
def calc_and_print_results(data: Point[np.ndarray], y_predict: np.ndarray) -> None:
    rss = np.sum((data.y - y_predict) ** 2)
    mean_y = np.mean(data.y)
    tss = np.sum((data.y - mean_y) ** 2)
    r2 = 1 - (rss / tss) if tss != 0 else 0

    print(f"R^2 для мультипликативной модели: {r2:.4f}")
