from typing import List

# Week 5: May 27th - May 31st

# <-------------------------------------------------- May 27th, 2024 -------------------------------------------------->
# 1608. Special Array With X Elements Greater Than or Equal X

# Given a non-negative integer array nums, find if it is special; that is, find a unique number 'x' where 'x' equals the
# count of elements in nums that are greater than or equal to 'x'.
# If such 'x' exists, return it, else return -1.


def specialArray1(nums: List[int]) -> int:
    """
    Finds a special number 'x' in the given integer array.

    This solution uses a brute-force approach to find the special number 'x'.
    The time complexity of this solution is O(n log(n)), where 'n' is the number of elements in the input array.
    :param nums: List of non-negative integers
    :return: Special number 'x' if it exists, otherwise -1
    """
    nums.sort()
    n = len(nums)

    for potential_special_x in range(n + 1):
        # Count how many numbers are greater than or equal to the potential special number 'x'
        numbers_greater_or_equal = sum(1 for num in nums if num >= potential_special_x)

        if numbers_greater_or_equal == potential_special_x:
            return potential_special_x

    return -1


nums = [3, 5]
print(specialArray1(nums))  # Output: 2

nums = [0, 0]
print(specialArray1(nums))  # Output: -1

nums = [0, 4, 3, 0, 4]
print(specialArray1(nums))  # Output: 3
