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
    result = 0
    letter_counts = defaultdict(int)

    for letter in s:
        letter_counts[letter] += 1

    for count in letter_counts.values():
        result += count * (count + 1) // 2

    # Alternative one-liner solution using Counter which effectively does the same thing:
    # return sum(count * (count + 1) // 2 for count in Counter(s).values())

    return result


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
arrays = [[2, 3, 6, 8], [1, 2, 3, 5, 6, 7, 10], [2, 3, 4, 6, 9]]
# longestCommonSubsequence1(arrays)  # Expected output: [2, 3, 6]
# longestCommonSubsequence2(arrays)  # Expected output: [2, 3, 6]

# Test cases for Week 2, June
s = "abcba"
numberOfSubstrings1(s)  # Expected output: 7
s = "abacad"
numberOfSubstrings1(s)  # Expected output: 9

# Test cases for Week 3, June

# Test cases for Week 4, June

# Test cases for Week 5, June
