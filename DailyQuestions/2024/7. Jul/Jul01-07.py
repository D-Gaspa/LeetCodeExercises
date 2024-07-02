# Week 1: July 1st - July 7th
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
    # Ensure nums1 is the shorter array for efficiency
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    nums1.sort()
    nums2.sort()

    intersection = []
    index1, index2 = 0, 0
    while index1 < len(nums1) and index2 < len(nums2):
        if nums1[index1] < nums2[index2]:
            index1 += 1
        elif nums1[index1] > nums2[index2]:
            index2 += 1
        else:
            intersection.append(nums1[index1])
            index1 += 1
            index2 += 1

    return intersection


def intersect2(nums1: List[int], nums2: List[int]) -> List[int]:
    # Ensure nums1 is the shorter array for efficiency
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    count_nums1 = Counter(nums1)

    intersection = []
    for num in nums2:
        if count_nums1[num] > 0:
            intersection.append(num)
            count_nums1[num] -= 1

    return intersection

# <-------------------------------------------------- July 3rd, 2024 -------------------------------------------------->
# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


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
intersect1(nums1=[4, 9, 5], nums2=[9, 4, 9, 8, 4])
intersect2(nums1=[4, 9, 5], nums2=[9, 4, 9, 8, 4])

# Test cases for July 3rd, 2024

# Test cases for July 4th, 2024

# Test cases for July 5th, 2024

# Test cases for July 6th, 2024

# Test cases for July 7th, 2024
