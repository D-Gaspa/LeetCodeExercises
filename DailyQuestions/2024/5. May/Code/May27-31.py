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
    """
    nums.sort()
    n = len(nums)

    for potential_special_x in range(n + 1):
        # Count how many numbers are greater than or equal to the potential special number 'x'
        numbers_greater_or_equal = sum(1 for num in nums if num >= potential_special_x)

        if numbers_greater_or_equal == potential_special_x:
            return potential_special_x

    return -1


# <-------------------------------------------------- May 28th, 2024 -------------------------------------------------->
# 1208. Get Equal Substrings Within a Budget

# Given two strings `s` and `t` of the same length and a budget `max_cost`, find the length of the longest substring
# of `s` you can transform into the corresponding substring of `t` by changing characters, where the cost of changing
# each character is the absolute difference of their ASCII values, and the total cost cannot exceed `max_cost`.


def equalSubstring(s: str, t: str, max_cost: int) -> int:
    """
    Finds the length of the longest substring of 's' that can be transformed into the corresponding substring of 't'
    within the given budget 'max_cost'.

    This solution uses a sliding window approach to find the longest substring.
    The time complexity of this solution is O(n), where 'n' is the length of the input strings.
    """
    n = len(s)
    cost = [abs(ord(s[i]) - ord(t[i])) for i in range(n)]  # Calculate the cost of changing each character

    start_index = 0
    total_cost = 0
    max_length = 0

    for end_index in range(n):  # Sliding window
        total_cost += cost[end_index]

        while total_cost > max_cost:
            total_cost -= cost[start_index]
            start_index += 1

        max_length = max(max_length, end_index - start_index + 1)

    return max_length


# <---------------------------------------------------- Test cases ---------------------------------------------------->

# Test cases for May 27th, 2024
nums = [3, 5]
print(specialArray1(nums))  # Output: 2

nums = [0, 0]
print(specialArray1(nums))  # Output: -1

nums = [0, 4, 3, 0, 4]
print(specialArray1(nums))  # Output: 3

# Test cases for May 28th, 2024
s = "abcd"
t = "bcdf"
max_cost = 3
print(equalSubstring(s, t, max_cost))  # Output: 3

s = "abcd"
t = "cdef"
max_cost = 3
print(equalSubstring(s, t, max_cost))  # Output: 1

s = "abcd"
t = "acde"
max_cost = 0
print(equalSubstring(s, t, max_cost))  # Output: 1
