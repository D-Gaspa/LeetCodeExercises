import math
from typing import List

from tabulate import tabulate

# Week 3: June 17th - June 23rd, 2024

# <------------------------------------------------- June 17th, 2024 ------------------------------------------------->
# 633. Sum of Square Numbers

# Given a non-negative integer `c`, decide whether there are two integers `a` and `b` such that `a^2 + b^2 = c`.


def judgeSquareSum1(c: int) -> bool:
    """
    Determines if a given non-negative integer 'c' can be expressed as the sum of squares of two integers 'a' and 'b'.

    The function uses a two-pointer technique starting from 0 and square root of 'c'.
    It then iteratively checks the sum of squares of the two pointers, 'start_index' and 'end_index'.
    If the sum is less than 'c', it increases the start_index to get a larger sum.
    If it's greater than 'c', it reduces the end_index to get a smaller sum.
    If it's equal to 'c', the function immediately returns True because it has found the pair of numbers.
    If no pair satisfying the condition is found after the loop, it returns False.
    This approach works because for every value of 'c', if there exist two numbers 'a' and 'b' such that a^2 + b^2 = c,
    then 'a' and 'b' must each be less than or equal to sqrt(c).

    The time complexity of this function is O(√c) because in the worst case,
    the while loop iterates up to the square root of c times.
    The space complexity is O(1) as it uses a constant amount of extra space.
    """
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
    index = 2
    while index * index <= c:
        divisors_count = 0
        if c % index == 0:
            while c % index == 0:
                divisors_count += 1
                c //= index
            if divisors_count % 2 and index % 4 == 3:
                return False
        index += 1
    return c % 4 != 3

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
# judgeSquareSum2(20)  # Expected output: True

# Test cases for June 18th, 2024

# Test cases for June 19th, 2024

# Test cases for June 20th, 2024

# Test cases for June 21st, 2024

# Test cases for June 22nd, 2024

# Test cases for June 23rd, 2024
