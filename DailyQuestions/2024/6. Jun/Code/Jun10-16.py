from pprint import pprint
from typing import List

from tabulate import tabulate

# Week 3: June 10th - June 16th, 2024

# <------------------------------------------------- June 10th, 2024 ------------------------------------------------->
# 1051. Height Checker

# Given an integer array heights representing the current order of students' heights in line,
# return the number of indices where heights[i] does not match the expected non-decreasing order of heights.


def heightChecker1(heights: List[int]) -> int:
    """
    Counts the number of students who are not standing in their expected height order.

    This function first creates a sorted copy of the input 'heights' list, representing the expected order.
    It then iterates through both the original and sorted lists, comparing corresponding elements at each index.
    If a mismatch is found (indicating a student out of order), the `unexpected_heights` counter is incremented.
    Finally, the function returns the total count of students not in their expected positions.

    The time complexity of this solution is O(n log n) due to the sorting operation,
    where n is the length of the 'heights' list.
    The space complexity is O(n) because a new list 'expected' is created to store the sorted heights.
    """
    expected = sorted(heights)
    unexpected_heights = 0

    for index in range(len(heights)):
        if heights[index] != expected[index]:
            unexpected_heights += 1
    return unexpected_heights


def heightChecker2(heights: List[int]) -> int:
    """
    Counts the number of students who are not standing in their expected height order.

    The function uses the Counting Sort algorithm which is an integer sorting algorithm that sorts arrays with integer
    keys in linear time.
    The function first creates a deepcopy `expected` of the `heights` list and applies counting sort on `expected`.
    Then, it compares each element in the `heights` and `expected` lists using zip().
    It increments a counter each time the compared elements are different.

    The time complexity of this solution is O(n + k), where n is the number of elements in the heights' list,
    and k is the range of values (max - min).
    The counting sort operation takes O(n + k) time: O(n) for counting and O(k) for reconstructing the sorted list.
    There are also two other O(n) operations: creating the expected list and comparing the elements.
    The space complexity is O(n + k) for storing the copied list and the counts in the dictionary.
    """

    def counting_sort(arr: List[int]) -> None:
        """Perform counting sort on the input array in-place."""
        min_val, max_val = min(arr), max(arr)

        # Create a dictionary to count occurrences of each value
        counts_map = {}
        for num in arr:
            if num in counts_map:
                counts_map[num] += 1
            else:
                counts_map[num] = 1

        # Reconstruct the array based on the counts
        index = 0
        for val in range(min_val, max_val + 1):
            if val in counts_map:
                count = counts_map[val]
                for _ in range(count):
                    arr[index] = val
                    index += 1

    expected = heights[:]
    counting_sort(expected)

    return sum(h1 != h2 for h1, h2 in zip(heights, expected))


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
heightChecker1(heights)  # Expected output: 3
heightChecker2(heights)  # Expected output: 3

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024
