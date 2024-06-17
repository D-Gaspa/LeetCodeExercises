import math
from typing import List

from tabulate import tabulate

# Week 3: June 17th - June 23rd, 2024

# <------------------------------------------------- June 17th, 2024 ------------------------------------------------->
# 633. Sum of Square Numbers

# Given a non-negative integer `c`, decide whether there are two integers `a` and `b` such that `a^2 + b^2 = c`.


def judgeSquareSum1(c: int) -> bool:
    start_index = 0
    end_index = int(math.sqrt(c))

    while start_index <= end_index:
        squares_sum = start_index * start_index + end_index * end_index  # a * a instead of a ** 2 because it's faster
        if squares_sum < c:
            start_index += 1
        elif squares_sum > c:
            end_index -= 1
        else:
            return True
    return False


def judgeSquareSum2(c: int) -> bool:
    pass

# <------------------------------------------------- June 18th, 2024 ------------------------------------------------->
# 2. Problem

# Description


def problem2_1():
    pass


def problem2_2():
    pass


# <------------------------------------------------- June 19th, 2024 ------------------------------------------------->
# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


# <------------------------------------------------- June 20th, 2024 ------------------------------------------------->
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <------------------------------------------------- June 21st, 2024 ------------------------------------------------->
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <------------------------------------------------- June 22nd, 2024 ------------------------------------------------->
# 6. Problem

# Description


def problem6_1():
    pass


def problem6_2():
    pass


# <------------------------------------------------- June 23rd, 2024 ------------------------------------------------->
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for June 17th, 2024
# judgeSquareSum1(20)  # Expected output: True

# Test cases for June 18th, 2024

# Test cases for June 19th, 2024

# Test cases for June 20th, 2024

# Test cases for June 21st, 2024

# Test cases for June 22nd, 2024

# Test cases for June 23rd, 2024
