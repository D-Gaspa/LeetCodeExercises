from pprint import pprint
from typing import List

from tabulate import tabulate


# Week 1: June 1st - June 2nd, 2024

# <-------------------------------------------------- June 1st, 2024 -------------------------------------------------->
# 3110. Score of a String

# The score of a string is defined as the sum of the absolute difference between the ASCII values
# of adjacent characters.
# Given a string s, return the score of s.


def scoreOfString(s: str) -> int:
    """
    Calculates the total score of a string based on absolute differences between adjacent character ASCII values.

    This function iterates through each pair of adjacent characters in the string,
    calculates the absolute difference between their ASCII values, and accumulates this into a total score.
    The final score represents the sum of these absolute differences,
    providing a measure of the string's overall character variation.

    The time complexity of this function is O(n), where n is the length of the string.
    This is because it performs a single iteration over the string's characters.
    The space complexity is O(1) as it uses a constant amount of extra space to store variables.
    """
    n = len(s)
    total_score = 0

    for index in range(n - 1):
        total_score += abs((ord(s[index]) - ord(s[index + 1])))

    # Alternatively, you can use pairwise from itertools to get the pairs of adjacent characters
    # s -> (s0, s1), (s1, s2), (s2, s3), ... This alternative approach would have similar time and space
    # complexities, but offers a more concise way of generating pairs

    return total_score


# <-------------------------------------------------- June 2nd, 2024 -------------------------------------------------->
# 2. Problem

# Description


def problem2_1():
    pass


def problem2_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for June 1st, 2024

# Test cases for June 2nd, 2024
