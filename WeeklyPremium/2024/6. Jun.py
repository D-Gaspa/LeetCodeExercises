import bisect
from collections import defaultdict
from typing import List

from tabulate import tabulate


# June, 2024
# <--------------------------------------------------- Week 1, June --------------------------------------------------->

# 1940. Longest Common Subsequence Between Sorted Arrays

# Given an array of integer arrays `arrays` where each `arrays[i]` is sorted in strictly increasing order,
# return an integer array representing the longest common subsequence between all the arrays.


def longestCommonSubsequence1(arrays: List[List[int]]) -> List[int]:
    """
    Finds the longest common subsequence present in the given list of sorted arrays.

    This function performs a preliminary check to identify the smallest array, then checks for numbers in this
    array across all other arrays in the list using a helper binary search function.

    The time complexity of this solution is O(n * m * log(k)), skewed towards the smallest size of the arrays.
    Where n is the number of elements in the smallest array, m is the total number of arrays,
    and k is the average length of the arrays.

    The space complexity is O(L), where L is the length of the longest common subsequence.
    """
    print("\n--- Input Parameters ---")
    print(f"\tarrays = {arrays}")

    def binary_search(array, num):
        """Helper function to perform binary search for 'num' in 'array'."""
        print(f"\t\tBinary searching for {num} in {array}")
        index = bisect.bisect_left(array, num)
        result = index != len(array) and array[index] == num
        print(f"\t\tFound: {result}")  # Print search result
        return result

    print("\n--- Identifying Smallest Array ---")
    smallest_array = min(arrays, key=len)
    arrays.remove(smallest_array)
    print(f"\tSmallest array: {smallest_array}")

    common_subsequences = []
    iteration_data = []

    print("\n--- Main Loop (Finding Common Elements) ---")
    for i, num in enumerate(smallest_array):
        print(f"\nIteration {i + 1}:")
        print(f"\tnum = {num}")
        # The line below should be kept intact for functionality reasons:
        if all(binary_search(array, num) for array in arrays):  # Check if 'num' is present in all other arrays
            common_subsequences.append(num)
            print(f"\t\tAdded to common_subsequences: {common_subsequences}")
        iteration_data.append([i + 1, num, common_subsequences.copy()])

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "num", "common_subsequences"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"longestCommonSubsequence1 = {common_subsequences}")
    return common_subsequences


def longestCommonSubsequence2(arrays: List[List[int]]) -> List[int]:
    """
    Finds the longest common subsequence present in the given list of sorted arrays.

    The main logic of the function is based on the two-pointer technique used in iterating through sorted lists
    for comparison.
    It iteratively finds the common elements of the first list (starting as the common subsequence) and the next ones.
    For each list, it runs through the elements of the current common subsequence and the new list side by side.
    If elements don't match, it advances the index of the smaller element.
    Every time a common value is found, it is added to the new_subsequence that ultimately replaces the current
    common_subsequence for next comparisons.

    The time complexity of this function is O(n * m), where n is the total number of lists,
    and m is the average size of these lists.
    This is because we are running through each list once, comparing and moving the pointers in a linear fashion.
    The space complexity is O(K), where K is the maximum length among the input arrays
    (used to store intermediate subsequences).
    """
    print("\n--- Input Parameters ---")
    print(f"\tarrays = {arrays}")

    common_subsequences = arrays[0]
    iteration_data = []  # To collect data for iteration summary

    print("\n--- Main Loop (Finding Common Subsequences) ---")
    for i, array in enumerate(arrays[1:], start=1):  # Start from the second array
        print(f"\nIteration {i}: Comparing with array {array}")

        new_subsequence = []
        array_index = 0
        common_subseq_index = 0
        array_length = len(array)
        common_subseq_length = len(common_subsequences)
        iteration_steps = []  # To collect data for each step within the iteration

        while common_subseq_index < common_subseq_length and array_index < array_length:
            print(
                f"\tComparing: common_subsequences[{common_subseq_index}] = {common_subsequences[common_subseq_index]}"
                f" vs. array[{array_index}] = {array[array_index]}")
            if array[array_index] == common_subsequences[common_subseq_index]:
                new_subsequence.append(array[array_index])
                print(f"\t\tMatch found! Added to new_subsequence: {new_subsequence}")
                print(f"\t\tAdvancing both pointers")
                common_subseq_index += 1
                array_index += 1
            elif array[array_index] < common_subsequences[common_subseq_index]:
                print(f"\t\tAdvancing array_index")
                array_index += 1
            else:
                print(f"\t\tAdvancing common_subseq_index")
                common_subseq_index += 1

            iteration_steps.append([common_subseq_index, array_index, new_subsequence.copy()])

        # Print Iteration Summary
        print(f"\n\t--- Iteration {i} Steps Summary ---")
        headers = ["common_subseq_index", "array_index", "new_subsequence"]
        print(tabulate(iteration_steps, headers=headers, tablefmt="fancy_grid"))

        common_subsequences = new_subsequence
        iteration_data.append([i, array, new_subsequence])  # Update summary for this iteration

    print("\n--- Iteration Summary (All Iterations) ---")
    headers = ["Iteration", "Array Compared", "Resulting Common Subsequence"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"longestCommonSubsequence2 = {common_subsequences}")
    return common_subsequences


# <--------------------------------------------------- Week 2, June --------------------------------------------------->
# 2083. Substrings That Begin and End With the Same Letter

# You are given a 0-indexed string 's' consisting of only lowercase English letters.
# Return the number of substrings in 's' that begin and end with the same character.
# A substring is a contiguous non-empty sequence of characters within a string.


def numberOfSubstrings1(s: str) -> int:
    """
    Calculates the number of substrings in a string 's' that begin and end with the same character.

    The function first creates a frequency dictionary 'letter_counts', which uses the defaultdict data structure to
    count the occurrences of each character in the string 's'.
    It then iterates over the values in the 'letter_counts' to calculate the total number of substrings that begin and
    end with the same character.
    The calculation count * (count + 1) // 2 derives from the formula to calculate the sum of the first 'n' integers,
    as each letter can form a substring with every other occurrence of the same letter, including itself.

    The time complexity of this solution is O(n), where 'n' is the length of the string 's'.
    This is because we iterate through the string once to count the occurrences of each character.
    The loop over the values in the 'letter_counts' dictionary has a constant number of iterations (26 at most).
    The space complexity is O(1) because the 'letter_counts' dictionary stores counts for at most 26
    different characters (lowercase English letters), so the space usage doesn't scale with the length of 's'.
    """
    print("\n--- Input Parameters ---")
    print(f"\ts: {s}")

    result = 0
    letter_counts = defaultdict(int)

    print("\n--- First Loop (Counting Letters) ---")
    iteration_data = []
    for letter in s:
        print(f"\tProcessing letter: {letter}")
        letter_counts[letter] += 1
        print(f"\t\t Updated letter_counts: {letter_counts}")
        iteration_data.append([letter, letter_counts.copy()])  # Store data for iteration summary

    print("\n--- Iteration Summary (Letter Counts) ---")
    print(tabulate(iteration_data, headers=["Letter", "Letter Counts"], tablefmt="fancy_grid"))

    print("\n--- Second Loop (Calculating Substrings) ---")
    for count in letter_counts.values():
        print(f"\tProcessing count: {count}")
        print(f"\t\tAdding ({count} * ({count} + 1) // 2 = {count * (count + 1) // 2}) to result ({result})")
        result += count * (count + 1) // 2
        print(f"\t\t Updated result: {result}")

    # Alternative one-liner solution (commented out):
    # return sum(count * (count + 1) // 2 for count in Counter(s).values())

    print("\n--- Function Returning ---")
    print(f"\tResult: {result}")
    return result


# <--------------------------------------------------- Week 3, June --------------------------------------------------->
# 1580. Put Boxes Into the Warehouse II

# Given two arrays of positive integers, `boxes` and `warehouse`, representing the heights of boxes and the heights of
# rooms in a warehouse respectively, determine the maximum number of boxes you can put into the warehouse.
# Boxes can be rearranged, pushed from either side, cannot be stacked, and if a room's height is less than a box's
# height, that box and all behind it will be stopped.

# Introductory problem:
# 1564. Put Boxes Into the Warehouse I

# We are asked to do the same thing as in the previous problem, but this time boxes can only be pushed from the left.


def maxBoxesInWarehouse1(boxes: List[int], warehouse: List[int]) -> int:
    """
    Calculates the maximum number of boxes that can be arranged in the warehouse given the
    heights of the boxes and the warehouse rooms from left to right only.

    Function sorts boxes and tries to fit the largest boxes first.
    If a box fits (its height is <= current warehouse room), the count increases, and it moves to the next room.
    If it doesn't fit, the box is discarded (continue to the next box).

    Time complexity is O(n log n) and space complexity is O(n), where `n` is the number of boxes, as we sort the boxes.
    """
    max_number = 0

    for box in sorted(boxes, reverse=True):
        if max_number < len(warehouse) and box <= warehouse[max_number]:
            max_number += 1

    return max_number


def maxBoxesInWarehouse2(boxes: List[int], warehouse: List[int]) -> int:
    max_number = 0
    left_index, right_index = 0, len(warehouse) - 1

    for box in sorted(boxes, reverse=True):
        if box <= warehouse[left_index]:
            max_number += 1
            left_index += 1
        elif box <= warehouse[right_index]:
            max_number += 1
            right_index -= 1
        if left_index == right_index:
            return max_number

    return max_number


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
arrays = [[2, 3, 6, 8], [1, 2, 3, 5, 6, 7, 10], [2, 3, 4, 6, 9]]
# longestCommonSubsequence1(arrays)  # Expected output: [2, 3, 6]
# longestCommonSubsequence2(arrays)  # Expected output: [2, 3, 6]

# Test cases for Week 2, June
s = "abacabd"
# numberOfSubstrings1(s)  # Expected output: 11

# Test cases for Week 3, June
maxBoxesInWarehouse1([4, 3, 4, 1], [5, 3, 3, 4, 1])  # Expected output: 3
maxBoxesInWarehouse1([1, 2, 2, 3, 4], [3, 4, 1, 2])  # Expected output: 3
maxBoxesInWarehouse1([1, 2, 3], [1, 2, 3, 4])  # Expected output: 1

maxBoxesInWarehouse2([1, 2, 2, 3, 4], [3, 4, 1, 2])  # Expected output: 4
maxBoxesInWarehouse2([3, 5, 5, 2], [2, 1, 3, 4, 5])  # Expected output: 3

# Test cases for Week 4, June

# Test cases for Week 5, June