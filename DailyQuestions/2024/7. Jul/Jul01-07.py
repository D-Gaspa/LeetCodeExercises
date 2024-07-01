# Week 1: July 1st - July 7th
from typing import List


# <-------------------------------------------------- July 1st, 2024 -------------------------------------------------->
# 1550. Three Consecutive Odds

# Given an integer `arr`, return `true` if there are three consecutive odd numbers in the array.
# Otherwise, return `false`.


def threeConsecutiveOdds1(arr: List[int]) -> bool:
    """
    Determines if there are three consecutive odd numbers in the given array.

    This function iterates through the array once, keeping track of the count of consecutive odd numbers encountered.
    It uses a single variable 'consecutive_odds' to maintain this count, resetting it to 0 whenever an even number is
    found. This approach is memory-efficient and allows for a single-pass solution.

    The time complexity of this solution is O(n), where `n` is the length of the input array, because it performs a
    single iteration through the array in the worst case. The space complexity is O(1) as it uses only a constant
    amount of extra space regardless of the input size.
    """
    if len(arr) < 3:
        return False

    consecutive_odds = 0
    for num in arr:
        if num % 2:
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
