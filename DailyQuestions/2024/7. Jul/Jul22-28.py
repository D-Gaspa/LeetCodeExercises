# Week 4: July 22th - July 28th, 2024
from typing import List
from collections import OrderedDict

# <------------------------------------------------ July 22th, 2024 ------------------------------------------------>
# 2418. Sort the People

# You are given two array of strings names, and heights that consists of distinct positive integers. Both arrays are of same length n.
# For each index i, names[i] and heights[i] denote the name and height of the ith person.
# Return the names array sorted in descending order based on the corresponding heights.


def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
    print("\n--- Input Parameters ---")
    print(f"\tnames = {names}")
    print(f"\theights = {heights}")

    number_of_people = len(names)
    print(f"\n--- Number of People ---")
    print(f"\tnumber_of_people = {number_of_people}")

    print("\n--- Creating Height-Name Map ---")
    height_to_name_map = dict(zip(heights, names))
    print(f"\theight_to_name_map = {height_to_name_map}")

    print("\n--- Sorting Heights in Descending Order ---")
    sorted_heights = sorted(heights, reverse=True)
    print(f"\tsorted_heights = {sorted_heights}")

    print("\n--- Creating Sorted Names List Based on Descending Heights ---")
    sorted_names = [height_to_name_map[height] for height in sorted_heights]
    print(f"\tsorted_names = {sorted_names}")

    return sorted_names



def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
    print("\n--- Input Parameters ---")
    print(f"\tnames = {names}")
    print(f"\theights = {heights}")

    number_of_people = len(names)
    print(f"\n--- Number of People ---")
    print(f"\tnumber_of_people = {number_of_people}")

    print("\n--- Creating Height-Name Map ---")
    height_to_name_map = OrderedDict()
    for height, name in zip(heights, names):
        height_to_name_map[height] = name
    print(f"\theight_to_name_map (before sorting) = {height_to_name_map}")

    print("\n--- Sorting Height-Name Map by Height in Descending Order ---")
    height_to_name_map = OrderedDict(
        sorted(height_to_name_map.items(), reverse=True)
    )
    print(f"\theight_to_name_map (after sorting) = {height_to_name_map}")

    print("\n--- Creating Sorted Names List Based on Descending Heights ---")
    sorted_names = list(height_to_name_map.values())
    print(f"\tsorted_names = {sorted_names}")

    return sorted_names



# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 2. Problem

# Description


def problem2_1():
    pass


def problem2_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 3. Problem

# Description


def problem3_1():
    pass


def problem3_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 4. Problem

# Description


def problem4_1():
    pass


def problem4_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 5. Problem

# Description


def problem5_1():
    pass


def problem5_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 6. Problem

# Description


def problem6_1():
    pass


def problem6_2():
    pass


# <------------------------------------------------ Month day th, 2024 ------------------------------------------------>
# 7. Problem

# Description


def problem7_1():
    pass


def problem7_2():
    pass

# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024

# Test cases for month day th, 2024
