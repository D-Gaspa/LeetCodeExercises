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
    print("\n--- Input Parameters ---")
    print(f"\ts = {s}")
    print(f"\tt = {t}")

    s_index = 0
    t_index = 0
    s_length = len(s)
    t_length = len(t)
    iterations = []

    print("\n--- Main Loop (Comparing 's' and 't') ---")
    while s_index < s_length and t_index < t_length:
        print(f"\nIteration of Main Loop: {s_index + 1}")
        print(f"\tCurrent 's_index': {s_index}")
        print(f"\tCurrent 't_index': {t_index}")
        match = False

        # Summary for this iteration
        iterations.append([s_index, t_index, s[s_index], t[t_index], ])

        # Condition check
        print(f"\tComparing s[{s_index}] = {s[s_index]} with t[{t_index}] = {t[t_index]}")
        if s[s_index] == t[t_index]:
            print("\t\tTrue Branch: s and t characters match")
            t_index += 1
            match = True
        else:
            print("\t\tFalse Branch: s and t characters don't match")
        s_index += 1

        # Add the match status to the summary
        iterations[-1].append("Yes" if match else "No")

    print("\n--- Iteration Summary (Current Indexes and Matched Characters) ---")
    headers = ["s_index", "t_index", "s[s_index]", "t[t_index]", "Match?"]
    print(tabulate(iterations, headers=headers, tablefmt="fancy_grid"))

    final_result = t_length - t_index

    print("\n--- Function return value ---")
    print(f"Final Result: t_length - t_index = {t_length} - {t_index} = {final_result}")

    return final_result


# <-------------------------------------------------- June 4th, 2024 -------------------------------------------------->
# 409 - Longest Palindrome

# Given a string 's', which consists of lowercase or uppercase letters, return the length of the longest palindrome
# that can be built with those letters.


def longestPalindrome(s: str) -> int:
    """
    Calculates the length of the longest palindrome that can be built with the characters in the input string 's'.

    First, the function counts the frequency of each character using the 'char_count' dictionary.
    Then, it iterates through the counts and if the count is even, it adds the count to the result.
    If the count is odd, it adds one less than the count to the result and sets 'odd_exists' flag to True.
    This is done because palindromes can have at most one character with an odd count (at the center of the palindrome);
    all other characters must occur an even number of times.
    Finally, if there was at least one character with an odd count,
    it adds 1 to the result, accounting for the possible center element in the palindrome.

    The total time complexity of this function is O(n) because it iterates over the string 's' once to count characters
    and iterates over every character frequency in 'char_count' once.
    The space complexity of this function is O(1) because the 'char_count' dictionary will at most contain entries
    equal to the number of different characters which are constant.
    """
    char_count = {}

    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    result = 0
    odd_exists = False

    for _, count in char_count.items():
        if count % 2 == 0:
            result += count
        else:
            result += count - 1
            odd_exists = True

    # If there was at least one character with an odd count, it can be used as the center of the palindrome
    if odd_exists:
        result += 1

    return result


def longestPalindrome2(s: str) -> int:
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
# appendCharacters1(s, t)  # Expected output: 4

# Test cases for june 4th, 2024
s_2 = "abccccdd"
# longestPalindrome(s_2)  # Expected output: 7

# Test cases for june 5th, 2024

# Test cases for june 6th, 2024

# Test cases for june 7th, 2024

# Test cases for june 8th, 2024

# Test cases for june 9th, 2024
