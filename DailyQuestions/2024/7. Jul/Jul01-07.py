# Week 1: July 1st - July 7th
from typing import List


# <-------------------------------------------------- July 1st, 2024 -------------------------------------------------->
# 1550. Three Consecutive Odds

# Given an integer `arr`, return `true` if there are three consecutive odd numbers in the array.
# Otherwise, return `false`.


def threeConsecutiveOdds1(arr: List[int]) -> bool:
    n = len(arr)
    if n < 3:
        return False

    consecutive_odds = 0
    for index in range(n):
        if arr[index] % 2:
            consecutive_odds += 1
        else:
            consecutive_odds = 0
        if consecutive_odds == 3:
            return True

    return False


# <-------------------------------------------------- July 2nd, 2024 -------------------------------------------------->
# 2. Problem

# Description


def problem2_1():
    pass


def problem2_2():
    pass


# <-------------------------------------------------- July 3rd, 2024 -------------------------------------------------->
# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


# <-------------------------------------------------- July 4th, 2024 -------------------------------------------------->
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <-------------------------------------------------- July 5th, 2024 -------------------------------------------------->
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <-------------------------------------------------- July 6th, 2024 -------------------------------------------------->
# 6. Problem

# Description


def problem6_1():
    pass


def problem6_2():
    pass


# <-------------------------------------------------- July 7th, 2024 -------------------------------------------------->
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for July 1st, 2024
# Expected output: False
print(threeConsecutiveOdds1(arr=[2, 6, 4, 1]))
# Expected output: True
print(threeConsecutiveOdds1(arr=[1, 2, 34, 3, 4, 5, 7, 23, 12]))

# Test cases for July 2nd, 2024

# Test cases for July 3rd, 2024

# Test cases for July 4th, 2024

# Test cases for July 5th, 2024

# Test cases for July 6th, 2024

# Test cases for July 7th, 2024
