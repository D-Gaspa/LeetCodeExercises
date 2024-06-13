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
    """
    Sorts arr1 such that elements are ordered as in arr2, with remaining elements in ascending order.

    This function uses a defaultdict to store the frequency of each element in arr1.
    It iterates through arr2, appending elements to the result list based on their frequency.
    Remaining elements not found in arr2 are sorted and appended at the end.

    The time complexity of this solution is O(n + m + r log r), where n is the length of arr1, m is the length of arr2,
    and r is the number of elements in arr1 that are not present in arr2.
    This is because we iterate through arr1 to count frequencies O(n), then iterate through arr2 to add elements
    O(m), and finally sort the remaining elements O(r log r).
    Here, r can vary from 0 to n, so the worst-case time complexity is O(n log n).
    The space complexity is O(n) to store the frequency counts in the hashmap and the result list.
    """
    print("\n--- Input Parameters ---")
    print(f"\tarr1 = {arr1}")
    print(f"\tarr2 = {arr2}")

    print("\n--- Building Frequency Dictionary ---")
    counts_dict = defaultdict(int)
    for num in arr1:
        counts_dict[num] += 1
        print(f"\tProcessed {num}, counts_dict: {dict(counts_dict)}")  # Show updated dictionary

    print("\n--- Appending Elements from arr2 ---")
    result = []
    for num in arr2:
        print(f"\n\tProcessing num {num} with count: {counts_dict[num]}")
        while counts_dict[num] > 0:
            result.append(num)
            counts_dict[num] -= 1
            print(f"\t\tAppended {num}, updated result: {result}, counts_dict[{num}]: {counts_dict[num]}")
    print("\n\tResult after processing arr2:", result)

    print("\n--- Collecting Remaining Elements Not in arr2 ---")
    remaining = []
    for num, count in counts_dict.items():
        if count > 0:  # Check if the count is greater than 0
            remaining.extend([num] * count)  # Append as many times as its count
            print(f"\t\tAdded {num} (count: {count}) to remaining: {remaining}")

    print("\n--- Sorting Remaining Elements ---")
    remaining.sort()  # In-place sort
    print(f"\tSorted remaining: {remaining}")
    result.extend(remaining)
    print(f"\tFinal Result: {result}")

    print("\n--- Function Returning ---")
    return result


def relativeSortArray2(arr1: List[int], arr2: List[int]) -> List[int]:
    """
    Sorts arr1 such that elements are ordered as in arr2, with remaining elements in ascending order.

    This function employs a counting sort strategy.
    It first determines the maximum value in arr1 to create a count array that can store frequencies of all elements
    After counting element occurrences in arr1,
    it constructs the sorted result by iterating over arr2 and appending elements based on their frequencies.
    Remaining elements not in arr2 are then added in ascending order.

    The time complexity of this solution is O(n + m + k), where n is the length of arr1, m is the length of arr2,
    and k is the range of values in arr1 (max value + 1).
    Finding the maximum value takes O(n) time, and counting frequencies takes O(n).
    Adding elements based on arr2 takes O(n + m) time, and adding remaining elements takes O(n + k) time.
    Hence, the overall time complexity is O(n + m + k).

    The space complexity is O(k) to store the count array.
    """
    print("\n--- Input Parameters ---")
    print(f"\tarr1 = {arr1}")
    print(f"\tarr2 = {arr2}")

    print("\n--- Finding Maximum Value in arr1 ---")
    max_val = max(arr1)
    print(f"\tMax value: {max_val}")

    print("\n--- Building Count Array ---")
    count = [0] * (max_val + 1)
    for num in arr1:
        count[num] += 1
        print(f"\tProcessed {num}, count: {count[num]}")  # Show an updated count array

    print("\n\tCount Array:", count)

    print("\n--- Appending Elements from arr2 Based on Count ---")
    result = []
    for num in arr2:
        print(f"\n\tProcessing num {num}")
        print(f"\t\tCount of {num}: {count[num]}")
        while count[num] > 0:
            result.append(num)
            count[num] -= 1
            print(f"\t\tAppended {num}, updated result: {result}, count[{num}]: {count[num]}")
    print("\n\tResult after processing arr2:", result)

    print("\n--- Appending Remaining Elements in Ascending Order ---")
    for num in range(max_val + 1):
        while count[num] > 0:
            result.append(num)
            count[num] -= 1
            print(f"\t\tAppended {num}, updated result: {result}, count[{num}]: {count[num]}")
    print(f"\n\tFinal Result: {result}")

    print("\n--- Function Returning ---")
    return result


# <------------------------------------------------- June 12th, 2024 ------------------------------------------------->
# 75. Sort Colors

# Given an array `nums` with `n` objects colored red, white, or blue, sort them in-place (without using built-in sort)
# so that objects of the same color are adjacent, with the colors in the order red, white, and blue.
# We will use the integers `0`, `1`, and `2` to represent the color red, white, and blue, respectively.


def sortColors1(nums: List[int]) -> None:
    """
    Sorts an array of integers containing only 0, 1, and 2 in-place.

    The function uses an algorithm similar to counting sort to do in-place sorting.
    It first counts the frequency of each color (0, 1, and 2 representing red, white, and blue respectively)
    and stores it in the 'color_counts' list.
    Subsequently, it rebuilds the `nums` list by placing the correct number of each color in the appropriate order.
    This approach avoids comparisons and leverages the limited value range for an efficient sorting process.

    The time complexity of this solution is O(n) due to two linear iterations over the input list `nums`:
    one for counting and another for reconstruction.
    The space complexity is O(1) as it uses a fixed-size list `color_counts` to store counts of three colors.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")  # Display the input list

    print("\n--- Main Loop (Counting Colors) ---")
    color_counts = [0, 0, 0]  # Initialize color counts
    iteration_data = []  # Collect data for iteration summary

    for i, num in enumerate(nums):
        print(f"\n\tIteration {i + 1}:")
        print(f"\t\tNumber: {num}")
        color_counts[num] += 1
        print(f"\t\tColor Counts (0=Red, 1=White, 2=Blue): {color_counts}")
        iteration_data.append([i + 1, num, color_counts.copy()])

    print("\n--- Iteration Summary (Color Counts) ---")
    headers = ["Iteration", "Number", "Color Counts"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Main Loop (Reconstructing Array) ---")
    index = 0
    for color in range(3):
        print(f"\n\tColor {color}:")
        for _ in range(color_counts[color]):
            print(f"\t\tPlacing color {color} at index {index}")
            nums[index] = color
            index += 1
            print(f"\t\tUpdated nums: {nums}")

    print("\n--- Function Returning ---")
    print(f"\tSorted nums: {nums}")


def sortColors2(nums: List[int]) -> None:
    """
    Sorts an array of integers containing only 0, 1, and 2 in-place.

    The function uses a variation of the three-way partitioning quicksort algorithm, often referred to as the
    Dutch national flag problem, to sort the array in-place.
    It maintains three pointers: 'left_index' to track the position to place the next '0' (red), 'current_index'
    for the currently evaluating number, and 'right_index' to place the next '2' (blue).
    If the current number is '0', it swaps this number with the number at 'left_index' position,
    then moves both 'left_index' and 'current_index' one step to the right.
    If it's '1', it leaves the number in place and just moves 'current_index'.
    If it's '2', it swaps this number with the number at 'right_index' position and decrements 'right_index'.

    The time complexity of this solution is O(n) because we perform a single pass over the list 'nums'.
    The space complexity is O(1) because we only used a few integer variables and didn't use any additional
    data structure that scales with the size of the input.
    """
    print("\n--- Input Parameters ---")
    print(f"\tnums = {nums}")  # Display an input list

    print("\n--- Main Loop (Dutch National Flag Algorithm) ---")
    left_index = 0  # Position to place the next '0'
    current_index = 0  # Current index being evaluated
    right_index = len(nums) - 1  # Position to place the next '2'
    iteration_data = [[left_index, current_index, right_index, nums.copy()]]

    while current_index <= right_index:
        print(f"\n\tWhile current_index ({current_index}) <= right_index ({right_index}):")
        print(f"\t\tleft_index = {left_index}, current_index = {current_index}, right_index = {right_index}")
        print(f"\t\tnums = {nums}")

        if nums[current_index] == 0:
            print("\t\tCondition: nums[current_index] == 0")
            print(f"\t\tSwapping 0 with element at left_index ({nums[left_index]})")
            print(f"\t\tIncrementing left_index {left_index} and current_index {current_index} by 1")
            nums[left_index], nums[current_index] = nums[current_index], nums[left_index]
            left_index += 1
            current_index += 1
        elif nums[current_index] == 1:
            print("\t\tCondition: nums[current_index] == 1")
            print("\t\tSkipping 1 (already in correct position)")
            print(f"\t\tIncrementing current_index {current_index} by 1")
            current_index += 1
        else:  # nums[current_index] == 2
            print("\t\tCondition: nums[current_index] == 2")
            print(f"\t\tSwapping 2 with element at right_index ({nums[right_index]})")
            print(f"\t\tDecrementing right_index {right_index} by 1")
            nums[current_index], nums[right_index] = nums[right_index], nums[current_index]
            right_index -= 1

        iteration_data.append([left_index, current_index, right_index, nums.copy()])

    print("\n--- Iteration Summary (Pointer Positions and Array State) ---")
    headers = ["left_index", "current_index", "right_index", "nums"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tSorted nums: {nums}")


# <------------------------------------------------- June 13th, 2024 ------------------------------------------------->
# 2037. Minimum Number of Moves to Seat Everyone

# Given two arrays seats and students of length n, representing the positions of n seats and n students in a room,
# you can adjust the position of each student by 1 unit any number of times.
# The task is to find the minimum number of moves required to seat all students such that no two students share the
# same seat, even if initially, multiple seats or students may occupy the same position.


def minMovesToSeat1(seats: List[int], students: List[int]) -> int:
    moves = 0
    seats.sort()
    students.sort()

    for seats, students in zip(seats, students):
        moves += abs(students - seats)

    return moves


def minMovesToSeat2(seats: List[int], students: List[int]) -> int:
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
nums = [2, 0, 2, 1, 1, 0]
# sortColors1(nums)  # Expected output: [0,0,1,1,2,2]
# sortColors2(nums)  # Expected output: [0,0,1,1,2,2]

# Test cases for month day th, 2024
seats = [4, 1, 5, 9]
students = [1, 3, 2, 6]
minMovesToSeat1(seats, students)  # Expected output: 7

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024
