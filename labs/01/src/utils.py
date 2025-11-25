import math
import random


def gen_vector(length: int) -> list[float]:
    return [random.randint(-10, 10) for _ in range(length)]


def strip_concatenate(x: list, y: list) -> list:
    if len(x) != len(y):
        raise ValueError("Wrong length")
    return [z for pair in zip(x, y) for z in pair]

# l inf-норма - максимальный элемент по модулю
def inf_norm(vector: list[float]) -> float:
    return max(abs(x) for x in vector)

# Обобщённая p-норма для любого p >= 1
def p_norm(vector: list[float], p: float) -> float:
    if p < 1:
        raise ValueError("p должно быть >= 1 для настоящей нормы")
    return sum(abs(x) ** p for x in vector) ** (1 / p)

def dot(x: list[float], y: list[float]) -> float:
    if len(x) != len(y):
        raise ValueError("Списки должны быть одинаковой длины")

    result = 0
    for i in range(len(x)):
        result += x[i] * y[i]
    return result
