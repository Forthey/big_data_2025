from __future__ import annotations

import random
from typing import Callable


class Vector[T]:
    def __init__(self, length: int):
        self.length: int = length
        self.__data: list[T] = [random.randint(-10, 10) for _ in range(self.length)]

    def __len__(self) -> int:
        return self.length

    def __str__(self) -> str:
        return f"Vector({",\t".join(map(str, self.__data))})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __index_assert(self, index: int):
        assert 0 <= index < self.length, IndexError(f"Wrong vector index: {index}")

    def __getitem__(self, index: int) -> T:
        self.__index_assert(index)

        return self.__data[index]
    
    def __setitem__(self, index: int, value: T):
        self.__index_assert(index)

        self.__data[index] = value
    
    def __iter__(self):
        _index: int = 0

        while _index < len(self):
            yield self[_index]
            _index += 1

    def sort(self, key: Callable[[T, T], bool] | None = None):
        self.__data.sort(key=key)

    @staticmethod
    def connect[K, V](vector1: Vector[K], vector2: Vector[V]) -> Vector[K | V]:
        length = max(len(vector1), len(vector2))
        
        result: Vector[K | V] = Vector[K | V](length)

        for index in range(length):
            result[index] = (vector2[index] if index % 2 else vector1[index])

        result.sort()

        return result

    def norm_1(self) -> T:
        result = 0

        for value in self.__data:
            result += abs(value)
        
        return result
    
    def norm_2(self) -> T:
        result = 0

        for value in self.__data:
            result += value ** 2
        
        return result ** 0.5
    
    def norm_inf(self) -> T:
        return max(map(abs, self.__data))
    
    def norm_w(self, w: Vector[float | int]) -> Vector[T]:
        result = 0

        for index in range(len(self)):
            result += w[index] * abs(self.__data[index])
        
        return result



def create_vecotrs[T](length: int) -> tuple[Vector[T], Vector[T]]:
    return Vector[T](length), Vector[T](length)


def task_2() -> tuple[Vector[int], Vector[int]]:
    print("TASK_2\n")
    x, y = create_vecotrs(10)
    print(f"{x=},\n{y=}")
    print()

    return x, y


def task_3(x: Vector[int], y: Vector[int]) -> Vector[int]:
    print("TASK_3\n")
    z = Vector.connect(x, y)
    print(f"{z=}")
    print()

    return z


def print_norm[T](name: str, vector: Vector[T]):
    print(f"{name}-vector norms:")
    print(f"l^1:\t{vector.norm_1():.3f}")
    print(f"l^2:\t{vector.norm_2():.3f}")
    print(f"l^inf:\t{vector.norm_inf():.3f}")
    print()


def task_4(x: Vector[int], y: Vector[int], z: Vector[int]):
    print("TASK_4\n")
    print_norm("x", x)
    print_norm("y", y)
    print_norm("z", z)
    print()


def factorial(number: int):
    result: int = 1

    for value in range(2, number + 1):
        result *= value
    
    return result


def task_6():
    print("TASK_5\n")
    vector: Vector[float] = Vector[float](5)

    print("Fill the vector X:")
    for index in range(len(vector)):
        vector[index] = float(input(f"[{index}]:\t"))

    print()
    
    print(f"min:\t{min(vector):.3f}")
    print(f"max:\t{max(vector):.3f}")
    print(f"sum:\t{sum(vector):.3f}")
    print(f"l^1:\t{vector.norm_1():.3f}")
    print(f"l^2:\t{vector.norm_2():.3f}")
    print(f"l^inf:\t{vector.norm_inf():.3f}")

    print()
    w_vector: Vector[float] = Vector[float](5)

    print("Fill the vector W:")
    for index in range(len(w_vector)):
        w_vector[index] = float(input(f"[{index}]:\t"))
    
    print()
    print(f"l^w:\t{vector.norm_w(w_vector)}")


def main():
    x, y = task_2()
    z = task_3(x, y)
    task_4(x, y, z)
    task_6()


if __name__ == "__main__":
    # 7 3 2 203 1
    # 0.2 0.3 0.4 0.01 0.1
    main()
