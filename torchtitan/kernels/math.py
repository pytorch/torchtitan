# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************


def ceil_divide(x: int, y: int) -> int:
    return (x + y - 1) // y


def check_power_of_2(n: int) -> bool:
    return n & (n - 1) == 0 and n != 0


def get_powers_of_2(start: int, end: int) -> list[int]:
    assert check_power_of_2(start), "start is not a power of 2"
    assert check_power_of_2(end), "end is not a power of 2"

    output = []
    n = start
    while n <= end:
        output.append(n)
        n = n << 1

    return output


def divide_if_divisible(dividend: int, divisor: int, msg: str = "") -> int:
    assert dividend % divisor == 0, msg
    return dividend // divisor


_POWERS_OF_2 = get_powers_of_2(1, 4294967296)


def get_next_power_of_2(x: int) -> int:
    for p in _POWERS_OF_2:
        if p >= x:
            return p

    raise ValueError(f"x ({x}) is bigger than the max allowable power of 2 ({p})")
