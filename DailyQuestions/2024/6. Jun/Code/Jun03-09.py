from pprint import pprint
from typing import List

from tabulate import tabulate

# Week 2: June 3rd - June 9th, 2024

# <-------------------------------------------------- June 3rd, 2024 -------------------------------------------------->
# 2486. Append Characters to String to Make Subsequence

# You are given two strings s and t consisting of only lowercase English letters.
# Return the minimum number of characters that need to be appended to the end of 's'
# so that 't' becomes a subsequence of s


def appendCharacters1(s: str, t: str) -> int:
    """
    Calculates the minimum number of characters from string 't'
    that must be appended to string 's' to make 't' a subsequence of 's'.

    This function iterates through both strings, comparing characters at corresponding positions.
    When a match is found, it advances in both strings; otherwise, it only moves forward in the first string.
    The function effectively checks if 't' is a subsequence of 's'
    (meaning 't' can be formed by deleting zero or more characters from 's').
    The result is the number of characters remaining in 't' after the comparison,
    indicating how many need to be appended.

    The function operates in O(n) time complexity,
    since each character of the string 's' and 't' is visited at most once.
    The space complexity is O(1) as the solution here does not require additional space that scales with input size.
    """

    s_index = 0
    t_index = 0

    s_length = len(s)
    t_length = len(t)

    while s_index < s_length and t_index < t_length:
        if s[s_index] == t[t_index]:
            t_index += 1
        s_index += 1

    # Return the count of remaining characters in 't' that needs to be appended to 's'
    return t_length - t_index


# <-------------------------------------------------- June 4th, 2024 -------------------------------------------------->
# 2. Problem

# Description


def problem2_1():
    pass


def problem2_2():
    pass


# <-------------------------------------------------- June 5th, 2024 -------------------------------------------------->
# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


# <-------------------------------------------------- June 6th, 2024 -------------------------------------------------->
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <-------------------------------------------------- June 7th, 2024 -------------------------------------------------->
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <-------------------------------------------------- June 8th, 2024 -------------------------------------------------->
# 6. Problem

# Description


def problem6_1():
    pass


def problem6_2():
    pass


# <-------------------------------------------------- June 9th, 2024 -------------------------------------------------->
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for june 3rd, 2024
s = "coaching"
t = "coding"
appendCharacters1(s, t)  # Expected output: 4

# Test cases for june 4th, 2024

# Test cases for june 5th, 2024

# Test cases for june 6th, 2024

# Test cases for june 7th, 2024

# Test cases for june 8th, 2024

# Test cases for june 9th, 2024
