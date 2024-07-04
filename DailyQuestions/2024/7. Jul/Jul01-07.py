# Week 1: July 1st - July 7th
import heapq

from collections import Counter
from typing import List

from tabulate import tabulate


# <-------------------------------------------------- July 1st, 2024 -------------------------------------------------->
# 1550. Three Consecutive Odds

# Given an integer `arr`, return `true` if there are three consecutive odd numbers in the array.
# Otherwise, return `false`.


def threeConsecutiveOdds1(arr):
    print("\n--- Input Parameters ---")
    print(f"\tarr = {arr}")

    print("\n--- Initialization ---")
    consecutive_odds = 0
    print(f"\tconsecutive_odds = {consecutive_odds}")

    print("\n--- Input Validation ---")
    if len(arr) < 3:
        print(f"\tArray length ({len(arr)}) is less than 3")
        print("\tReturning False")
        return False
    else:
        print(f"\tArray length ({len(arr)}) is valid")

    print("\n--- Main Loop ---")
    iteration_data = []
    process_finished = False
    for i, num in enumerate(arr):
        print(f"\n--- Element {i + 1}/{len(arr)} ---")
        print(f"\tCurrent number: {num}")
        print(f"\tconsecutive_odds at start: {consecutive_odds}")

        print("\tChecking if number is odd:")
        if num % 2:
            print(f"\t\t{num} is odd (remainder when divided by 2 is {num % 2})")
            consecutive_odds += 1
            print(f"\t\tIncrementing consecutive_odds to {consecutive_odds}")
        else:
            print(f"\t\t{num} is even (remainder when divided by 2 is {num % 2})")
            consecutive_odds = 0
            print(f"\t\tResetting consecutive_odds to {consecutive_odds}")

        print("\tChecking if three consecutive odds found:")
        if consecutive_odds == 3:
            print("\t\tThree consecutive odds found!")
            print("\t\tReturning True")
            process_finished = True
            iteration_data.append([i + 1, num, "Odd" if num % 2 else "Even", consecutive_odds])
            break
        else:
            print(f"\t\tNot yet three consecutive odds (current count: {consecutive_odds})")

        iteration_data.append([i + 1, num, "Odd" if num % 2 else "Even", consecutive_odds])

    print("\n--- Iteration Summary ---")
    headers = ["Element", "Number", "Odd/Even", "Consecutive Odds"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    if process_finished:
        return True

    print("\n--- Function Returning ---")
    print("\tThree consecutive odds not found in the entire array")
    print("\tReturning False")
    return False


# <-------------------------------------------------- July 2nd, 2024 -------------------------------------------------->
# 350. Intersection of Two Arrays II

# Given two integer arrays `nums1` and `nums2`, return an array of their intersection.
# Each element in the result must appear as many times as it shows in both arrays,
# and you may return the result in any order.


def intersect1(nums1: List[int], nums2: List[int]) -> List[int]:
    print("\n--- Input Parameters ---")
    print(f"\tnums1 = {nums1}")
    print(f"\tnums2 = {nums2}")

    print("\n--- Initialization ---")
    if len(nums1) > len(nums2):
        print("\tSwapping nums1 and nums2 to ensure nums1 is shorter")
        nums1, nums2 = nums2, nums1
    print(f"\tAfter potential swap: nums1 = {nums1}, nums2 = {nums2}")

    print("\nSorting both arrays:")
    nums1.sort()
    nums2.sort()
    print(f"\tSorted nums1 = {nums1}")
    print(f"\tSorted nums2 = {nums2}")

    intersection = []
    index1, index2 = 0, 0
    print(f"\nInitial state: index1 = {index1}, index2 = {index2}, intersection = {intersection}")

    print("\n--- Main Loop ---")
    iteration_data = []
    while index1 < len(nums1) and index2 < len(nums2):
        print(f"\n--- Iteration {len(iteration_data) + 1} ---")
        print(f"While loop condition: index1 ({index1}) < len(nums1) ({len(nums1)}) "
              f"and index2 ({index2}) < len(nums2) ({len(nums2)})")
        print(f"\tCurrent state: index1 = {index1}, index2 = {index2}")
        print(f"\tComparing nums1[{index1}] = {nums1[index1]} and nums2[{index2}] = {nums2[index2]}")

        if nums1[index1] < nums2[index2]:
            print(f"\t\tnums1[{index1}] < nums2[{index2}], incrementing index1")
            index1 += 1
            action = "Increment index1"
        elif nums1[index1] > nums2[index2]:
            print(f"\t\tnums1[{index1}] > nums2[{index2}], incrementing index2")
            index2 += 1
            action = "Increment index2"
        else:
            print(f"\t\tnums1[{index1}] == nums2[{index2}], adding to intersection and incrementing both indices")
            intersection.append(nums1[index1])
            index1 += 1
            index2 += 1
            action = "Add to intersection, increment both"

        iteration_data.append([len(iteration_data) + 1, index1, index2, nums1[index1 - 1] if index1 > 0 else None,
                               nums2[index2 - 1] if index2 > 0 else None, action, intersection.copy()])

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Index1", "Index2", "nums1 value", "nums2 value", "Action", "Current Intersection"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal intersection: {intersection}")
    return intersection


def intersect2(nums1: List[int], nums2: List[int]) -> List[int]:
    print("\n--- Input Parameters ---")
    print(f"\tnums1 = {nums1}")
    print(f"\tnums2 = {nums2}")

    print("\n--- Initialization ---")
    if len(nums1) > len(nums2):
        print("\tSwapping nums1 and nums2 to"
              " ensure nums1 is shorter")
        nums1, nums2 = nums2, nums1
    print(f"\tAfter potential swap: nums1 = {nums1}, nums2 = {nums2}")

    count_nums1 = Counter(nums1)
    print("\nCreating frequency map of nums1:")
    print(tabulate(count_nums1.items(), headers=["Element", "Frequency"], tablefmt="fancy_grid"))

    intersection = []
    print(f"\nInitial state: intersection = {intersection}")

    print("\n--- Main Loop ---")
    iteration_data = []
    for i, num in enumerate(nums2):
        print(f"\n--- Iteration {i + 1}/{len(nums2)} ---")
        print(f"\tProcessing num = {num}")
        print(f"\tCurrent count in frequency map: {count_nums1[num]}")

        if count_nums1[num] > 0:
            print(f"\t\tnum {num} found in frequency map with count > 0")
            intersection.append(num)
            count_nums1[num] -= 1
            action = f"Add {num} to intersection, decrement count"
        else:
            print(f"\t\tnum {num} not found in frequency map or count is 0")
            action = "No action"

        iteration_data.append([i + 1, num, count_nums1[num], action, intersection.copy()])

        print(f"\tUpdated frequency map for {num}: {count_nums1[num]}")
        print(f"\tCurrent intersection: {intersection}")

    print("\n--- Iteration Summary ---")
    headers = ["Iteration", "Number", "Count after action", "Action", "Current Intersection"]
    print(tabulate(iteration_data, headers=headers, tablefmt="fancy_grid"))

    print("\n--- Function Returning ---")
    print(f"\tFinal intersection: {intersection}")
    return intersection


# <-------------------------------------------------- July 3rd, 2024 -------------------------------------------------->
# 1509. Minimum Difference Between Largest and Smallest Value in Three Moves

# Return the minimum difference between the largest and smallest value of `nums` after performing at most three moves.
# In one move, you can choose one element of `nums` and change it to any value.


def minDifference1(nums: List[int]) -> int:
    """
    Calculates the minimum difference between the largest and smallest values in nums
    after performing at most three moves.

    This function uses a greedy approach to find the optimal solution. It recognizes that to minimize the difference,
    we should focus on changing either the smallest or largest elements (or a combination of both). The function
    first handles the edge case where the list has 42 or fewer elements.
    Then, it extracts the 4 smallest and 4 largest elements, as these are the only ones that could potentially affect
    the result given the constraint of at most three moves.
    It then considers all possible combinations of changing three elements and returns the minimum difference achieved.

    The time complexity of this solution is O(n log 4), which simplifies to O(n), where n is the length of nums.
    This is because nsmallest and nlargest operations take O(n log k) time, where k is 4 in this case.
    The space complexity is O(1) as we only store a constant number of elements (8 in total) regardless of input size.
    """
    # If we have 4 or fewer elements, we can make all elements equal
    if len(nums) <= 4:
        return 0

    smallest_four = heapq.nsmallest(4, nums)
    largest_four = heapq.nlargest(4, nums)

    # Check all possible combinations of 3 moves
    return min(
        largest_four[0] - smallest_four[3],  # Change 3 smallest
        largest_four[1] - smallest_four[2],  # Change 2 smallest, 1 largest
        largest_four[2] - smallest_four[1],  # Change 1 smallest, 2 largest
        largest_four[3] - smallest_four[0]  # Change 3 largest
    )


# <-------------------------------------------------- July 4th, 2024 -------------------------------------------------->
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <-------------------------------------------------- July 5th, 2024 -------------------------------------------------->
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <-------------------------------------------------- July 6th, 2024 -------------------------------------------------->
# 6. Problem

# Description


def problem6_1():
    pass


def problem6_2():
    pass


# <-------------------------------------------------- July 7th, 2024 -------------------------------------------------->
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for July 1st, 2024
# Expected output: True
# threeConsecutiveOdds1(arr=[1, 2, 34, 3, 4, 5, 7, 23, 12])

# Test cases for July 2nd, 2024
# Expected output: [4, 9]
# intersect1(nums1=[4, 9, 5], nums2=[9, 4, 9, 8, 4])
# intersect2(nums1=[4, 9, 5], nums2=[9, 4, 9, 8, 4])

# Test cases for July 3rd, 2024
# Expected output: 4
print(minDifference1(nums=[6, 5, 0, 7, 10, 4, 8, 21]))

# Test cases for July 4th, 2024

# Test cases for July 5th, 2024

# Test cases for July 6th, 2024

# Test cases for July 7th, 2024
