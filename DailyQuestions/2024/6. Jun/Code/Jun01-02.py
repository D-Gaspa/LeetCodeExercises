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
    print("\n--- Input Parameters ---")
    print(f"\ts = '{s}'")  # Display the input string

    n = len(s)
    print(f"\tn (string length) = {n}")  # Display calculated length

    total_score = 0
    print(f"\ttotal_score (initialized) = {total_score}")  # Show initial score

    print("\n--- Main Loop (Calculating Character Pair Scores) ---")
    iteration_data = []  # Collect data for iteration summary
    for index in range(n - 1):
        print(f"\n\tIteration {index + 1}:")
        char1 = s[index]
        char2 = s[index + 1]
        print(f"\t\tCharacters: '{char1}' and '{char2}'")

        char1_ascii = ord(char1)
        char2_ascii = ord(char2)
        print(f"\t\tASCII Values: {char1_ascii} and {char2_ascii}")

        pair_score = abs(char1_ascii - char2_ascii)
        print(f"\t\tPair Score: {pair_score}")

        total_score += pair_score
        print(f"\t\tUpdated Total Score: {total_score}")

        # Store data for iteration summary
        iteration_data.append([index + 1, char1, char2, char1_ascii, char2_ascii, pair_score, total_score])

    print("\n--- Iteration Summary (Character Pair Scores) ---")
    headers = ["Iteration", "Char 1", "Char 2", "ASCII 1", "ASCII 2", "Pair Score", "Total Score"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"Final Total Score: {total_score}")
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
test_input_1 = "hello"
# scoreOfString(test_input_1)  # Expected output: 13

# Test cases for June 2nd, 2024
