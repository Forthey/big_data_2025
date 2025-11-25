from utils import gen_vector, strip_concatenate, p_norm, inf_norm, dot


# 2
def generate_vectors() -> tuple[list[float], list[float]]:
    x, y = gen_vector(10), gen_vector(10)
    print(f"\n{x=},\n{y=}")

    return x, y

# 3
def create_vector_z(x: list[float], y: list[float]) -> list[float]:
    z = strip_concatenate(x, y)
    print(f"\n{z=}")

    return z

# 4
def print_norms(name: str, vector: list[float]):
    print(f"\nКлассические нормы для {name}:")
    print(f"l^1:\t{p_norm(vector, 1):.3f}")
    print(f"l^2:\t{p_norm(vector, 2):.3f}")
    print(f"l^inf:\t{inf_norm(vector):.3f}")

# 4
def print_vectors_norms(x: list[float], y: list[float], z: list[float]):
    print_norms("x", x)
    print_norms("y", y)
    print_norms("z", z)

# 6
def factorial(x: int):
    result: int = 1

    for sub_x in range(2, x + 1):
        result *= sub_x
    
    return result

# 7
def fill_interactive():
    vector: list[float] = [0.0] * 5

    print("\nfill vector x:")
    for index in range(len(vector)):
        vector[index] = float(input(f"{index}:\t"))
    
    print(f"\nmin: {min(vector):.3f}")
    print(f"max: {max(vector):.3f}")
    print(f"sum: {sum(vector):.3f}")
    print_norms("x", vector)

    w_vector: list[float] = [0.0] * 5

    print("\nfill vector w:")
    for index in range(len(w_vector)):
        w_vector[index] = float(input(f"[{index}]:\t"))
    print(f"l^w:\t{dot(vector, w_vector)}")


def main():
    x, y = generate_vectors()
    z = create_vector_z(x, y)
    z.sort()
    print_vectors_norms(x, y, z)
    fill_interactive()


if __name__ == "__main__":
    # 7 3 2 203 1
    # 0.2 0.3 0.4 0.01 0.1
    main()
