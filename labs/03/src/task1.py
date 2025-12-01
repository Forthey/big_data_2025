from dataclasses import dataclass
from typing import Any

# Required PyQt6 (UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown pyplot.show())
import matplotlib.pyplot as pyplot
import numpy as np


@dataclass
class PronyApproxResult:
    # Амплитуда
    amp:        np.ndarray[complex]
    # Частота
    omega:    np.ndarray[complex]
    # Начальная фаза
    phi:      np.ndarray[complex]
    # Коэффициент затухания
    decay_coef:   np.ndarray[complex]
    # Период дескретизации сигнала
    dt:    complex

FIELD_TO_STRING: dict[Any, str] = {
    "amp":      "Амплитуда",
    "omega":  "Частота",
    "phi":    "Начальная фаза",
    "decay_coef": "Коэффициент затухания",
    "dt":  "Период дескретизации сигнала",
}


def prony(x: np.ndarray, m: int, delta_t: float = 1.0) -> PronyApproxResult:
    n = len(x)
    rows = n - m

    X = np.zeros((rows, m), dtype=complex)
    y = np.zeros(rows,      dtype=complex)
    x = np.array(x,         dtype=complex)

    for index_row in range(rows):
        for index_col in range(m):
            X[index_row, index_col] = x[m + index_row - index_col - 1]
        y[index_row] = x[m + index_row]

    a = np.linalg.lstsq(X, -y, rcond=None)[0]

    coeffs = np.concatenate(([1.], a))
    z = np.roots(coeffs)
    decay_coef = np.log(np.abs(z)) / delta_t
    omega = np.atan(np.imag(z), np.real(z)) / (2 * np.pi * delta_t)

    v = np.zeros((n, m), dtype=complex)

    for k in range(n):
        for r in range(m):
            v[k, r] = z[r] ** k

    h = np.linalg.lstsq(v, x, rcond=None)[0]
    amp = np.abs(h)
    phi_i = np.atan(np.imag(h), np.real(h))

    return PronyApproxResult(amp, omega, phi_i, decay_coef, delta_t)


def get_model_row_from(data: PronyApproxResult, n: int, m: int) -> np.ndarray[complex]:
    result: np.ndarray[complex] = np.zeros((n, ), dtype=complex)

    for result_index in range(n):
        for sum_index in range(m):
            h_i: np.ndarray[complex] = data.amp[sum_index] * np.exp(1j * data.phi[sum_index])
            z_i: np.ndarray[complex] = np.exp((data.decay_coef[sum_index] + 2j * np.pi * data.omega[sum_index]) * data.dt)
            
            result[result_index] += h_i * z_i ** result_index
    
    return result


def get_model_row(n: int = 200, h: float = 0.02) -> np.ndarray[complex]:
    i = np.arange(1, n + 1)
    x = np.zeros(n)

    for k in range(1, 4):
        x += k * np.exp(-h * i / k) * np.cos(4 * np.pi * h * i * k + np.pi / k)
    
    return x


def print_graph(x_original: np.ndarray[complex], x_restored: np.ndarray[complex], n: int):
    indexes = np.arange(1, n + 1)

    pyplot.figure(figsize=(15, 10))
    pyplot.plot(indexes, x_original, label='Оригинальный',      color="black")
    pyplot.plot(indexes, x_restored, label='Восстановленный',   color="red")
    
    pyplot.legend()

    pyplot.title('Оригинальный и восстановленный модельный ряд $x_i$')
    pyplot.xlabel('i')
    pyplot.ylabel('$x_i$')

    pyplot.savefig("task1.png")


def compare_rows(x_original: np.ndarray[complex], x_restored: np.ndarray[complex]):
    return np.mean((x_original - x_restored) ** 2)


if __name__ == "__main__":
    n: int = 200

    model_row = get_model_row(n)
    print(f"Первые 12 элементов изначального ряда:\n{model_row[:12]}")

    m: int = 3
    data: PronyApproxResult = prony(model_row, m)

    print("\nРезультат аппроксимации:")
    for key, value in data.__dict__.items():
        value_output: str = value if isinstance(value, float) else ",\t".join(map(lambda item: f"{item:.3f}", value))
        print(f"{FIELD_TO_STRING[key]:<40}{value_output}")

    restored_row = np.real(get_model_row_from(data, n, m))
    print(f"\nПервые 12 элементов восстановленного ряда:\n{restored_row[:12]}")

    print(f"\nMSE: {compare_rows(model_row, restored_row)}")

    print_graph(model_row, restored_row, n)
