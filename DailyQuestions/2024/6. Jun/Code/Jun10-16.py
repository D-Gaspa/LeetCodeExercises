from collections import defaultdict
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
    print("\n--- Input Parameters ---")
    print(f"\theights = {heights}")

    # Sorting Heights
    print("\n--- Sorting Heights ---")
    expected = sorted(heights)
    print(f"\texpected (Sorted Heights) = {expected}")

    # Comparing Heights and Counting Mismatches
    print("\n--- Comparing Heights and Counting Mismatches ---")
    unexpected_heights = 0
    iteration_data = []

    for index in range(len(heights)):
        print(f"\n\tIteration {index + 1}:")
        print(f"\t\theights[{index}] = {heights[index]}, expected[{index}] = {expected[index]}")

        if heights[index] != expected[index]:
            unexpected_heights += 1
            print(f"\t\tMismatch found, incrementing counter (unexpected_heights = {unexpected_heights})")

        iteration_data.append([index + 1, heights[index], expected[index], unexpected_heights])

    # Display Iteration Summary Table
    print("\n--- Iteration Summary (Comparison Results) ---")
    headers = ["Iteration", "heights[index]", "expected[index]", "Total Mismatches"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    # Returning the Result
    print("\n--- Function Returning ---")
    print(f"\tUnexpected heights: {unexpected_heights}")
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
    print("\n--- Input Parameters ---")
    print(f"\theights = {heights}")

    # Counting Sort (Inner Function)
    def counting_sort(arr: List[int]) -> None:
        """Perform counting sort on the input array in-place."""
        print("\n--- Counting Sort (Inner Function) ---")
        min_val, max_val = min(arr), max(arr)
        print(f"\tMinimum Value: {min_val}, Maximum Value: {max_val}")

        # Building Count Dictionary
        print("\n\t--- Building Count Dictionary ---")
        counts_map = {}
        for num in arr:
            print(f"\t\tProcessing number: {num}")
            counts_map[num] = counts_map.get(num, 0) + 1
            print(f"\t\tUpdated counts_map: {counts_map}")

        # Reconstructing the Array
        print("\n\t--- Reconstructing the Array ---")
        index = 0
        reconstruction_data = []
        for val in range(min_val, max_val + 1):
            print(f"\t\tProcessing value: {val}")
            if val in counts_map:
                count = counts_map[val]
                print(f"\t\t\tCount of {val}: {count}")
                for _ in range(count):
                    arr[index] = val
                    index += 1
                    print(f"\t\t\tAdded {val} to the array at index {index - 1}")
                    reconstruction_data.append([val, index, arr.copy()])
            else:
                print(f"\t\t\tValue {val} not found in counts_map, skipping.")

        print("\n\t--- Reconstruction Summary ---")
        headers = ["Value", "Index", "Array (After Insertion)"]
        print(tabulate(reconstruction_data, headers=headers, tablefmt="fancy_grid"))

    # Copying and Sorting 'heights'
    print("\n--- Copying and Sorting Heights ---")
    expected = heights[:]
    print(f"\texpected (Before Sort) = {expected}")
    counting_sort(expected)
    print(f"\texpected (After Sort) = {expected}")

    # Comparing Heights and Counting Mismatches
    print("\n--- Comparing Heights and Counting Mismatches ---")
    unexpected_heights = 0
    comparison_data = []

    for h1, h2 in zip(heights, expected):
        print(f"\tComparing: {h1} (original) vs. {h2} (expected)")
        if h1 != h2:
            unexpected_heights += 1
            print("\t\tMismatch found, incrementing unexpected_heights counter.")
        comparison_data.append([h1, h2, unexpected_heights])

    print("\n--- Comparison Summary ---")
    headers = ["heights[i]", "expected[i]", "Total Mismatches"]
    print(tabulate(comparison_data, headers=headers, tablefmt="fancy_grid"))

    # Returning the Result
    print("\n--- Function Returning ---")
    print(f"\tUnexpected heights: {unexpected_heights}")
    return unexpected_heights


# <------------------------------------------------- June 11th, 2024 ------------------------------------------------->
# 1122. Relative Sort Array

# Given two arrays `arr1` and `arr2`, where `arr2` elements are distinct, and all elements in `arr2` are also in `arr1`.
# Sort the elements of `arr1` such that the relative ordering of items in `arr1` is the same as in `arr2`.
# Elements that do not appear in `arr2` should be placed at the end of `arr1` in ascending order.


def relativeSortArray1(arr1: List[int], arr2: List[int]) -> List[int]:
    counts_dict = defaultdict(int)
    for num in arr1:
        counts_dict[num] += 1

    # Add elements as per relative order
    result = []
    for num in arr2:
        for _ in range(counts_dict[num]):
            result.append(num)
            counts_dict[num] -= 1

    # Add remaining elements in ascending order
    remaining = []
    for num, count in counts_dict.items():
        for _ in range(count):
            remaining.append(num)

    result.extend(sorted(remaining))

    return result


def relativeSortArray2(arr1: List[int], arr2: List[int]) -> List[int]:
    max_val = max(arr1)
    count = [0] * (max_val + 1)

    for num in arr1:
        count[num] += 1

    # Add elements as per relative order
    result = []
    for num in arr2:
        while count[num] > 0:
            result.append(num)
            count[num] -= 1

    # Add remaining elements in ascending order
    for num in range(max_val + 1):
        while count[num] > 0:
            result.append(num)
            count[num] -= 1

    return result


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
# heightChecker1(heights)  # Expected output: 3
# heightChecker2(heights)  # Expected output: 3

# Test cases for month day th, 2024
arr1 = [2, 3, 1, 3, 2, 4, 6, 7, 9, 2, 19]
arr2 = [2, 1, 4, 3, 9, 6]
# relativeSortArray1(arr1, arr2)  # Expected output: [2,2,2,1,4,3,3,9,6,7,19]
# relativeSortArray2(arr1, arr2)  # Expected output: [2,2,2,1,4,3,3,9,6,7,19]

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024
