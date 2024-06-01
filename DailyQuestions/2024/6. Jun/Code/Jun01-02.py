from pprint import pprint
from typing import List

from tabulate import tabulate

# Week 1: June 1st - June 2nd, 2024

# <-------------------------------------------------- June 1st, 2024 -------------------------------------------------->
# 3110. Score of a String

# Given a string s, the score of the string is the sum of the scores of all of its substrings.
# The score of a substring is the number of 'a' characters in the substring, multiplied by the length of the substring.
# Return the score of the string s.


def scoreOfString(s: str) -> int:
    n = len(s)
    total_score = 0

    for index in range(n - 1):
        total_score += abs((ord(s[index]) - ord(s[index + 1])))

    return total_score


# <-------------------------------------------------- June 2nd, 2024 -------------------------------------------------->
# 2. Problem

# Description


def problem2_1():
    pass


def problem2_2():
    pass


