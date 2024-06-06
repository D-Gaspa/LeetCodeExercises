import bisect
from typing import List


# June, 2024

# <--------------------------------------------------- Week 1, June --------------------------------------------------->
# 1940. Longest Common Subsequence Between Sorted Arrays

# Given an array of integer arrays `arrays` where each `arrays[i]` is sorted in strictly increasing order,
# return an integer array representing the longest common subsequence between all the arrays.


def longestCommonSubsequence1(arrays: List[List[int]]) -> List[int]:
    """
    Finds the longest common subsequence present in the given list of sorted arrays.

    The function uses a binary search helper function in its main logic as an efficient way to check the existence
    of an element from the first array in all other given arrays.
    It iterates over every number in the first array, and checks if that number is present in every other array using
    the helper binary search function.
    If it's found in all arrays, it's added to the result list `common_subsequences`.

    The time complexity of this solution is O(n * m * log(k)), where n is the number of elements in the first array,
    m is the total number of arrays, and k is the average length of the arrays.
    We perform binary search (with time complexity O(log(k))) for each element in the first array (n times),
    across all arrays (m times).

    The space complexity is O(L), where L is the length of the longest common subsequence.
    This accounts for the space required to store the final list of common subsequences.
    """

    def binary_search(array, num):
        """Helper function to perform binary search for 'num' in 'array'."""
        index = bisect.bisect_left(array, num)
        if index != len(array) and array[index] == num:  # Check if 'num' exists at the insertion point
            return True
        else:
            return False

    common_subsequences = []

    for num in arrays[0]:
        if all(binary_search(array, num) for array in arrays[1:]):  # Check if 'num' is present in all other arrays
            common_subsequences.append(num)

    return common_subsequences


def longestCommonSubsequence2(arrays: List[List[int]]) -> List[int]:
    pass


# <--------------------------------------------------- Week 2, June --------------------------------------------------->
# 2. Problem

# Description


def problem2_1():
    pass


def problem2_2():
    pass


# <--------------------------------------------------- Week 3, June --------------------------------------------------->
# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


# <--------------------------------------------------- Week 4, June --------------------------------------------------->
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <--------------------------------------------------- Week 5, June --------------------------------------------------->
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for Week 1, June
arrays = [[1, 3, 4], [1, 4, 7, 9]]
# longestCommonSubsequence1(arrays)  # Expected output: [1, 4]
# longestCommonSubsequence2(arrays)  # Expected output: [1, 4]

# Test cases for Week 2, June

# Test cases for Week 3, June

# Test cases for Week 4, June

# Test cases for Week 5, June
