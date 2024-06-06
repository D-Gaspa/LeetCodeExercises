import bisect
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
    print("\n--- Input Parameters ---")
    print(f"\tarrays = {arrays}")

    def binary_search(array, num):
        """Helper function to perform binary search for 'num' in 'array'."""
        print(f"\t\tBinary searching for {num} in {array}")
        index = bisect.bisect_left(array, num)
        result = index != len(array) and array[index] == num
        print(f"\t\tFound: {result}")  # Print search result
        return result

    common_subsequences = []
    iteration_data = []  # To collect data for iteration summary

    print("\n--- Main Loop (Finding Common Elements) ---")
    for i, num in enumerate(arrays[0]):
        print(f"\nIteration {i + 1}:")
        print(f"\tnum = {num}")

        if all(binary_search(array, num) for array in arrays[1:]):
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
                f"\tComparing: common_subsequences[{common_subseq_index}] = {common_subsequences[common_subseq_index]} vs. array[{array_index}] = {array[array_index]}")
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
arrays = [[2, 3, 6, 8], [1, 2, 3, 5, 6, 7, 10], [2, 3, 4, 6, 9]]
# longestCommonSubsequence1(arrays)  # Expected output: [2, 3, 6]
# longestCommonSubsequence2(arrays)  # Expected output: [2, 3, 6]

# Test cases for Week 2, June

# Test cases for Week 3, June

# Test cases for Week 4, June

# Test cases for Week 5, June
