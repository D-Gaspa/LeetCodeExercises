from pprint import pprint
from typing import List

from tabulate import tabulate

# Week 3: June 10th - June 16th, 2024

# <------------------------------------------------- June 10th, 2024 ------------------------------------------------->
# 1051. Height Checker

# Given an integer array heights representing the current order of students' heights in line,
# return the number of indices where heights[i] does not match the expected non-decreasing order of heights.


def heightChecker1(heights: List[int]) -> int:
    expected = sorted(heights)
    unexpected_heights = 0

    for index in range(len(heights)):
        if heights[index] != expected[index]:
            unexpected_heights += 1
    return unexpected_heights


# <------------------------------------------------- June 11th, 2024 ------------------------------------------------->
# 2. Problem

# Description


def problem2_1():
    pass


def problem2_2():
    pass


# <------------------------------------------------- June 12th, 2024 ------------------------------------------------->
# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


# <------------------------------------------------- June 13th, 2024 ------------------------------------------------->
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <------------------------------------------------- June 14th, 2024 ------------------------------------------------->
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <------------------------------------------------- June 15th, 2024 ------------------------------------------------->
# 6. Problem

# Description


def problem6_1():
    pass


def problem6_2():
    pass


# <------------------------------------------------- June 16th, 2024 ------------------------------------------------->
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for month day th, 2024
heights = [1, 1, 4, 2, 1, 3]
# heightChecker1(heights))  # Expected output: 3

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024
